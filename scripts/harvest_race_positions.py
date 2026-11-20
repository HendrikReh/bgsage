# SPDX-License-Identifier: MPL-2.0
"""
Harvest pure-race positions for re-training the PureRace NN on rollout targets.

Why: the PureRace net (``sl_s9_purerace``, shared by every model since Stage 6)
is the one clear gap to GNUbg on the Benchmark PR sample. With the bearoff DB
(how the engine plays, and the scorer's default since PR #5), stage11 scores
0.55 / 0.57 at 1-ply / 2-ply on the 979 sampled race decisions against
GNUbg's 0.39 / 0.37. The DB settles the home-board bearoffs (0.09, GNUbg
0.11); the whole gap is in the 608 races OUTSIDE it (0.83 / 0.86 against
0.57 / 0.52), and there 2-ply is no better than 1-ply — so measure a new net
at both. `scripts/race_gap_analysis.py` and `scripts/race_net_swap.py`
reproduce and compare (the latter takes `--net "name=file:hidden"`; Stage 5's
200h race net already scores 0.64 at 2-ply there, against a Stage-5-biased
reference). The GNUbg race corpus the net was trained on is not on this
machine, so the new targets are our own rollouts.

What it does: plays N 1-ply self-play money games in lockstep through
``batch_checker_play`` (production model; every race position goes to the
shared pure-race net whatever the model). Contact moves are perturbed (with
probability ``--eps-contact`` a random move from the top 4) so the races that
follow are varied; race moves are perturbed less (``--eps-race``, top 3).
For every race decision the top ``--candidates`` post-move boards are kept —
so the net sees the alternatives it has to rank, not just the moves it
would play — except boards the bearoff DB answers exactly, finished games,
and every board of the Benchmark PR and money benchmark decisions (both
orientations). A game stops once both sides are in the bearoff DB.

Every 10th game goes to the holdout benchmark; boards in both sets are
dropped from train. Boards are post-move, the side that just moved positive,
which is what ``scripts/modal_rollout.py`` and the SL trainers expect.

Output: data/purerace-sage-train-data and data/purerace-sage-benchmark-data
(26 ints per line, shuffled), for

    python scripts/modal_rollout.py data/purerace-sage-train-data ...

Usage:
    python scripts/harvest_race_positions.py --games 20000
"""

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
from neongammon import STARTING_BOARD, batch_checker_play, check_game_over, flip_board, is_race  # noqa: E402
from neongammon.weights import bearoff_db_path  # noqa: E402

_DATA = _PROJECT_ROOT / "data"


def excluded_boards() -> set[tuple[int, ...]]:
    """Benchmark PR + money benchmark decision and candidate boards, both orientations."""
    boards: list[list[int]] = []
    pr = _DATA / "benchmark_pr" / "decisions.json"
    if pr.exists():
        for d in json.load(pr.open())["decisions"]:
            boards.append(d["board"])
            boards.extend(d["candidates"])
    for name in ("money_benchmark", "pasko_money_benchmark"):
        for fname, opener in (("benchmark.json.gz", gzip.open), ("benchmark.json", open)):
            path = _DATA / name / fname
            if not path.exists():
                continue
            with opener(path, "rt") as f:
                for d in json.load(f).get("decisions", []):
                    boards.append(d["board"])
                    for m in d.get("moves", []) + d.get("candidates", []):
                        boards.append(m["board"] if isinstance(m, dict) else m)
            break
    ex = set()
    for b in boards:
        ex.add(tuple(b))
        ex.add(tuple(flip_board(list(b))))
    return ex


def play(n_games: int, seed: int, n_candidates: int, eps_contact: float, eps_race: float,
         batch: int, db, excluded: set) -> tuple[list, list, dict]:
    """Return (train boards, benchmark boards, stats)."""
    train: dict[tuple[int, ...], None] = {}
    bench: dict[tuple[int, ...], None] = {}
    stats = {"race_decisions": 0, "skipped_bearoff": 0, "skipped_excluded": 0, "games": 0}

    for g0 in range(0, n_games, batch):
        games = []
        for g in range(g0, min(g0 + batch, n_games)):
            rng = random.Random(f"{seed}:race:{g}")
            d1, d2 = rng.sample(range(1, 7), 2)  # opening roll: no doubles
            games.append({"id": g, "rng": rng, "board": list(STARTING_BOARD), "dice": (d1, d2)})
        while games:
            res = batch_checker_play(
                [{"board": s["board"], "die1": s["dice"][0], "die2": s["dice"][1],
                  "cube_value": 1, "cube_owner": "centered"} for s in games],
                eval_level="1ply")
            alive = []
            for s, r in zip(games, res):
                rng, pre = s["rng"], s["board"]
                moves = r.moves
                if not moves:  # dance: board unchanged
                    post = pre
                else:
                    race = is_race(pre)
                    eps, top = (eps_race, 3) if race else (eps_contact, 4)
                    if rng.random() < eps:
                        post = list(moves[rng.randrange(min(top, len(moves)))].board)
                    else:
                        post = list(moves[0].board)
                    if race:
                        stats["race_decisions"] += 1
                        sink = bench if s["id"] % 10 == 0 else train
                        for m in moves[:n_candidates]:
                            b = list(m.board)
                            if check_game_over(b) != 0:
                                continue
                            if db is not None and db.is_bearoff(b):
                                stats["skipped_bearoff"] += 1
                                continue
                            k = tuple(b)
                            if k in excluded:
                                stats["skipped_excluded"] += 1
                                continue
                            sink[k] = None
                if check_game_over(post) != 0:
                    stats["games"] += 1
                    continue
                nxt = flip_board(post)
                if db is not None and db.is_bearoff(nxt):
                    stats["games"] += 1  # the DB takes it from here
                    continue
                s["board"] = nxt
                s["dice"] = (rng.randint(1, 6), rng.randint(1, 6))
                alive.append(s)
            games = alive
        print(f"  {min(g0 + batch, n_games)}/{n_games} games: {len(train)} train, "
              f"{len(bench)} benchmark boards", flush=True)

    bench_keys = set(bench)
    train_list = [b for b in train if b not in bench_keys]
    return train_list, list(bench), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--candidates", type=int, default=3,
                    help="top-N post-move boards kept per race decision")
    ap.add_argument("--eps-contact", type=float, default=0.2)
    ap.add_argument("--eps-race", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4000, help="games played in lockstep")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-bench", type=int, default=0)
    ap.add_argument("--prefix", default="purerace-sage")
    args = ap.parse_args()

    bgbot_cpp.init_escape_tables()
    db = None
    path = bearoff_db_path()
    if path:
        db = bgbot_cpp.BearoffDB()
        if not db.load(path):
            db = None
    if db is None:
        print("WARNING: no bearoff DB; bearoff boards are kept", flush=True)

    excluded = excluded_boards()
    print(f"exclusion set: {len(excluded)} boards", flush=True)
    t0 = time.time()
    train, bench, stats = play(args.games, args.seed, args.candidates, args.eps_contact,
                               args.eps_race, args.batch, db, excluded)
    print(f"{stats} in {time.time() - t0:.0f} s", flush=True)

    rng = random.Random(args.seed)
    for name, boards, limit in (("train", train, args.limit_train),
                                ("benchmark", bench, args.limit_bench)):
        rng.shuffle(boards)
        if limit:
            boards = boards[:limit]
        out = _DATA / f"{args.prefix}-{name}-data"
        with out.open("w") as f:
            for b in boards:
                f.write(" ".join(str(x) for x in b) + "\n")
        print(f"wrote {len(boards)} boards -> {out.relative_to(_PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
