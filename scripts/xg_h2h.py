#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Head-to-head money games, Sage against eXtreme Gammon, without an XG API.

XG cannot be driven programmatically on the Mac (it runs under CrossOver), but
its Batch Analyze can grade any position we write as a one-decision ``.xg``
game. This script turns that into real games by replaying to a fixed point:

1. Sage plays its own seat. For XG's seat it uses XG's recorded answer when it
   has one, and otherwise PREDICTS XG's decision with Sage and keeps playing.
2. Every predicted XG decision is written as a one-decision game; you run XG's
   Batch Analyze on the round's folder once.
3. The next ``round`` reads XG's answers back and replays every game from the
   start. Where XG agreed with the prediction the game simply goes further;
   where it did not, XG's move replaces the prediction and the rest of the
   game is played again, with new predictions for the next round.

A game is finished when every XG decision in it is XG's own answer, so the
finished games are exactly what a live match would have produced: the dice are
drawn per (game pair, turn number), never from the moves, so a changed move
cannot change any later roll.

Design rules (the ones upstream's retired replay runner broke):

* Standard opening: the opening roll is never a double; the side with the
  higher die moves first with both dice; no cube decision before it.
* Every XG decision is certified: XG's #1 checker play (validated against the
  legal moves) and XG's cube recommendation (double iff FlagDouble > 0, take
  iff DT <= DP), never a fallback to Sage. Forced moves and dances need no
  decision. Records XG leaves unanalysed stay pending.
* No game is ever skipped or dropped; ``report`` refuses to score an
  incomplete set unless asked to, and then says so.
* Duplicate dice: game pair ``p`` is played twice on the same dice with the
  seats swapped (``p####a``: Sage in seat 0, ``p####b``: Sage in seat 1).

Rules: money game, Jacoby ON, Beaver OFF (no beavers by either side; the XG
files' header says so too), unlimited cube.

Variance reduction (``report --luck``): each roll's luck, from Sage's own
2-ply analytics (``BgBotAnalyzer.roll_luck``), times the cube value is
subtracted from the result. Every roll's luck has expectation zero whatever
evaluator computes it, so the adjusted mean is unbiased; it only lowers the
variance.

Workflow::

    python scripts/xg_h2h.py init --pairs 50 --seed 1 --sage-level 3ply --dir data/xg_h2h/pilot
    python scripts/xg_h2h.py round --dir data/xg_h2h/pilot
      -> XG: File -> Batch Analyze on data/xg_h2h/pilot/round_001, the level you
         chose for the whole run, "Save games after analyze" ON
    python scripts/xg_h2h.py round --dir data/xg_h2h/pilot      # repeat until done
    python scripts/xg_h2h.py report --dir data/xg_h2h/pilot --luck

``round --simulate-xg 2ply`` lets a Sage level stand in for XG (no files, all
rounds at once) to test the machinery and measure rounds; use a separate
``--dir`` for that, ``init`` records which kind of run a directory is.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import export_folder_benchmark_xg as fx  # noqa: E402
from bgsage import (STARTING_BOARD, BgBotAnalyzer, check_game_over, flip_board,  # noqa: E402
                    possible_moves, xg_file as xf)
from export_pasko_benchmark_xg import _templates, derive_half_moves, write_xg  # noqa: E402

DEFAULT_TEMPLATES = (_PROJECT_ROOT / "data" / "xg_h2h" / "template.xg",
                     _PROJECT_ROOT / "data" / "benchmark_pr_xg" / "template.xg")
JACOBY, BEAVER = True, False
PER_SHARD = 200
MAX_TURNS = 2000
SIDECAR = ".sidecar.jsonl"


# ---------------------------------------------------------------------------
# Keys, dice, small helpers
# ---------------------------------------------------------------------------


def _b(board) -> str:
    return ",".join(str(int(x)) for x in board)


def cube_key(board, cv: int, owner: str) -> str:
    return f"C|{_b(board)}|{cv}|{owner}"


def checker_key(board, cv: int, owner: str, d1: int, d2: int) -> str:
    return f"M|{_b(board)}|{cv}|{owner}|{max(d1, d2)}{min(d1, d2)}"


def dice_for(seed: int, pair: int, turn: int) -> tuple[int, int]:
    """The roll of turn ``turn`` of game pair ``pair``; turn 0 is the opening
    roll (seat 0's die, seat 1's die), never a double."""
    rng = random.Random(f"xg-h2h:{seed}:{pair}:{turn}")
    while True:
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        if turn > 0 or d1 != d2:
            return d1, d2


def legal_posts(board, d1: int, d2: int) -> list[tuple[int, ...]]:
    seen: dict[tuple[int, ...], None] = {}
    for p in possible_moves(board, d1, d2):
        seen.setdefault(tuple(int(x) for x in p), None)
    return list(seen)


def swap_owner(owner: str) -> str:
    return {"player": "opponent", "opponent": "player"}.get(owner, owner)


def game_ids(n_pairs: int) -> list[tuple[str, int, int]]:
    """``(game id, pair, Sage's seat)`` for every game of the run."""
    return [(f"p{p:04d}{'ab'[s]}", p, s) for p in range(n_pairs) for s in (0, 1)]


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


# ---------------------------------------------------------------------------
# Sage as a decision oracle (own seat, predictions, stand-in XG), cached
# ---------------------------------------------------------------------------


class SageOracle:
    """Sage's decisions at one level, memoised on disk by decision key."""

    def __init__(self, level: str, cache_path: Path):
        self.level = level
        self.cache_path = cache_path
        self.cache = {r["key"]: r["answer"] for r in _read_jsonl(cache_path)}
        self._analyzer = None
        self.computed = 0

    @property
    def analyzer(self) -> BgBotAnalyzer:
        if self._analyzer is None:
            self._analyzer = BgBotAnalyzer(eval_level=self.level)
        return self._analyzer

    def _store(self, key: str, answer: dict) -> dict:
        self.cache[key] = answer
        self.computed += 1
        _append_jsonl(self.cache_path, {"key": key, "answer": answer})
        return answer

    def cube(self, board, cv: int, owner: str) -> dict:
        key = cube_key(board, cv, owner)
        if key not in self.cache:
            r = self.analyzer.cube_action(list(board), cv, owner, jacoby=JACOBY, beaver=BEAVER)
            self._store(key, {"double": bool(r.should_double), "take": bool(r.should_take)})
        return self.cache[key]

    def checker(self, board, cv: int, owner: str, d1: int, d2: int) -> dict:
        key = checker_key(board, cv, owner, d1, d2)
        if key not in self.cache:
            r = self.analyzer.checker_play(list(board), d1, d2, cube_value=cv, cube_owner=owner,
                                           jacoby=JACOBY, beaver=BEAVER)
            self._store(key, {"board": [int(x) for x in r.moves[0].board]})
        return self.cache[key]


# ---------------------------------------------------------------------------
# XG's answers
# ---------------------------------------------------------------------------


class XgAnswers:
    """XG's certified answers by decision key (``answers.jsonl``)."""

    def __init__(self, path: Path):
        self.path = path
        self.rows = {}
        self.conflicts: list[dict] = []
        for r in _read_jsonl(path):
            self.rows.setdefault(r["key"], r)

    def get(self, key: str) -> dict | None:
        r = self.rows.get(key)
        return None if r is None else r["answer"]

    def add(self, key: str, answer: dict, meta: dict) -> bool:
        """Record an answer; a different answer for a known key is a conflict
        (XG disagreeing with itself) and is reported, never overwritten."""
        old = self.rows.get(key)
        if old is not None:
            if old["answer"] != answer:
                self.conflicts.append({"key": key, "old": old, "new": {"answer": answer, **meta}})
            return False
        row = {"key": key, "answer": answer, **meta}
        self.rows[key] = row
        _append_jsonl(self.path, row)
        return True


# ---------------------------------------------------------------------------
# Playing one game against what is known so far
# ---------------------------------------------------------------------------


def play_game(seed: int, pair: int, sage_seat: int, sage: SageOracle, xg: XgAnswers,
              lookahead: int = 0) -> dict:
    """Play a game as far as the known XG answers allow, predicting the rest.

    Returns ``status`` ("done" when every XG decision was XG's own answer,
    else "pending"), the ``events``, the XG ``queries`` still unanswered (in
    play order) and, when the game reached its end, ``winner_seat``, ``points``
    and ``result`` (points from Sage's side). With ``lookahead`` > 0 the game
    stops after that many unanswered XG decisions.
    """
    events: list[dict] = []
    queries: list[dict] = []
    d1, d2 = dice_for(seed, pair, 0)
    mover = 0 if d1 > d2 else 1
    board = list(STARTING_BOARD)
    cv, owner, t = 1, "centered", 0

    def xg_or_predict(key: str, predict, query: dict) -> tuple[dict, str]:
        ans = xg.get(key)
        if ans is not None:
            return ans, "xg"
        queries.append(query)
        return predict(), "pred"

    def finish(winner: int, points: int, how: str) -> dict:
        return {"status": "pending" if queries else "done", "events": events, "queries": queries,
                "winner_seat": winner, "points": points, "how": how,
                "result": points if winner == sage_seat else -points, "turns": t}

    while t < MAX_TURNS:
        # --- cube action (never before the opening move) ---
        if t > 0 and owner in ("centered", "player"):
            ck = cube_key(board, cv, owner)
            q = {"kind": "cube", "key": ck, "board": list(board), "cv": cv, "owner": owner}
            if mover == sage_seat:
                ans, src = sage.cube(board, cv, owner), "sage"
            else:
                ans, src = xg_or_predict(ck, lambda: sage.cube(board, cv, owner), q)
            events.append({"t": t, "seat": mover, "kind": "offer", "double": ans["double"],
                           "src": src, "key": ck, "cv": cv, "owner": owner, "board": list(board)})
            if ans["double"]:
                responder = 1 - mover
                if responder == sage_seat:
                    resp, rsrc = sage.cube(board, cv, owner), "sage"
                else:
                    resp, rsrc = xg_or_predict(ck, lambda: sage.cube(board, cv, owner), q)
                events.append({"t": t, "seat": responder, "kind": "response",
                               "take": resp["take"], "src": rsrc, "key": ck, "cv": cv,
                               "owner": owner, "board": list(board)})
                if not resp["take"]:
                    return finish(mover, cv, "pass")
                cv, owner = cv * 2, "opponent"
            if lookahead and len(queries) >= lookahead:
                return {"status": "pending", "events": events, "queries": queries}

        # --- the roll and the checker play ---
        d1, d2 = dice_for(seed, pair, t) if t > 0 else (d1, d2)
        legal = legal_posts(board, d1, d2)
        if len(legal) <= 1:
            post, src, key = (list(legal[0]) if legal else list(board)), "forced", None
        else:
            key = checker_key(board, cv, owner, d1, d2)
            if mover == sage_seat:
                ans, src = sage.checker(board, cv, owner, d1, d2), "sage"
            else:
                q = {"kind": "checker", "key": key, "board": list(board), "cv": cv,
                     "owner": owner, "dice": [d1, d2], "opening": t == 0}
                ans, src = xg_or_predict(key, lambda: sage.checker(board, cv, owner, d1, d2), q)
                q["predicted"] = sage.checker(board, cv, owner, d1, d2)["board"]
            post = list(ans["board"])
            if tuple(post) not in set(legal):
                raise RuntimeError(f"{src} answer for {key} is not a legal move: {post}")
        events.append({"t": t, "seat": mover, "kind": "move", "dice": [d1, d2], "board": list(board),
                       "post": post, "cv": cv, "owner": owner, "src": src, "key": key})
        over = check_game_over(post)
        if over > 0:
            mult = 1 if (JACOBY and owner == "centered") else over
            return finish(mover, cv * mult, {1: "single", 2: "gammon", 3: "backgammon"}[over])
        board, owner, mover, t = list(flip_board(post)), swap_owner(owner), 1 - mover, t + 1
        if lookahead and len(queries) >= lookahead:
            return {"status": "pending", "events": events, "queries": queries}

    raise RuntimeError(f"game pair {pair} seat {sage_seat}: no result after {MAX_TURNS} turns")


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


class Run:
    def __init__(self, d: Path):
        self.dir = d
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            raise SystemExit(f"{d}: not a run directory (no config.json); run `init` first")
        self.cfg = json.loads(cfg_path.read_text())
        self.sage = SageOracle(self.cfg["sage_level"], d / "sage_cache.jsonl")
        self.xg = XgAnswers(d / "answers.jsonl")

    @property
    def games(self):
        return game_ids(self.cfg["pairs"])

    def round_dirs(self) -> list[Path]:
        return sorted(p for p in self.dir.glob("round_*") if p.is_dir())

    def play_all(self, lookahead: int = 0) -> dict[str, dict]:
        return {gid: play_game(self.cfg["seed"], pair, seat, self.sage, self.xg, lookahead)
                for gid, pair, seat in self.games}


def cmd_init(args) -> None:
    d = args.dir
    if (d / "config.json").exists():
        raise SystemExit(f"{d} already holds a run; pick a new --dir")
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"pairs": args.pairs, "seed": args.seed, "sage_level": args.sage_level,
           "opponent": f"standin:{args.standin}" if args.standin else "xg",
           "jacoby": JACOBY, "beaver": BEAVER, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"{d}: {args.pairs} pairs = {2 * args.pairs} games, Sage {args.sage_level} against "
          f"{cfg['opponent']}, seed {args.seed}")


# --- harvesting XG's analysis ------------------------------------------------


def harvest(run: Run) -> dict:
    """Read every analysed record of every round folder into ``answers.jsonl``."""
    st = Counter()
    levels = Counter()
    for rd in run.round_dirs():
        for shard in sorted(rd.glob("shard_*.xg")):
            rows = _read_jsonl(Path(str(shard) + SIDECAR))
            tx = xf.XgArchive.load(shard).get("temp.xg")
            games = fx.iter_games(tx)
            if len(games) != len(rows):
                raise SystemExit(f"{shard}: {len(games)} games but {len(rows)} sidecar rows")
            for row, g in zip(rows, games):
                meta = {"round": rd.name, "shard": shard.name, "game_index": row["game_index"]}
                if row.get("cube_key"):
                    st["cube_records"] += 1
                    pick, rec = fx.extract_cube_pick(tx, g["cube"][0])
                    if pick is None:
                        st["cube_unanalysed"] += 1
                    else:
                        levels[f"cube level {rec['level']}"] += 1
                        if run.xg.add(row["cube_key"], {"double": bool(pick["should_double"]),
                                                        "take": bool(pick["should_take"])},
                                      {**meta, "level": rec["level"],
                                       "flag_double": rec["flag_double"],
                                       "nd_dt_dp": [rec["equity_nd"], rec["equity_dt"],
                                                    rec["equity_dp"]]}):
                            st["cube_new"] += 1
                if row.get("checker_key"):
                    st["checker_records"] += 1
                    pick, rec = fx.extract_checker_pick(tx, g["move"][0])
                    if pick is None:
                        st["checker_unanalysed"] += 1
                        continue
                    d1, d2 = row["dice"]
                    if tuple(pick) not in set(legal_posts(row["board"], d1, d2)):
                        raise SystemExit(f"{shard} game {row['game_index']}: XG's #1 {pick} is not "
                                         f"a legal move of the exported position (encoding error)")
                    levels[f"checker level {rec['moves'][0]['level']}"] += 1
                    if run.xg.add(row["checker_key"], {"board": [int(x) for x in pick]},
                                  {**meta, "level": rec["moves"][0]["level"],
                                   "analyze_m": rec["analyze_m"]}):
                        st["checker_new"] += 1
    st["conflicts"] = len(run.xg.conflicts)
    return {"counts": dict(st), "levels": dict(levels)}


# --- writing a round's queries -------------------------------------------------


def plan_records(queries: list[dict]) -> list[dict]:
    """One-decision games for a round's queries, deduplicated.

    A checker query whose mover can double is written as "no double" + the
    roll, so XG analyses its cube decision as well; a separate cube query for
    the same position and cube is then dropped. Pure cube queries (XG's takes,
    and offers predicted to be doubles) get a fixed continuation roll."""
    by_key: dict[str, dict] = {}
    for q in queries:
        by_key.setdefault(q["key"], q)
    covered = set()
    recs = []
    for q in by_key.values():
        if q["kind"] != "checker":
            continue
        access = q["owner"] in ("centered", "player") and not q["opening"]
        ck = cube_key(q["board"], q["cv"], q["owner"]) if access else None
        if ck:
            covered.add(ck)
        d1, d2 = q["dice"]
        post = list(q["predicted"])
        recs.append({"board": q["board"], "cv": q["cv"], "owner": q["owner"], "dice": [d1, d2],
                     "post": post, "half_moves": derive_half_moves(q["board"], post, d1, d2),
                     "double": fx.NO_DOUBLE if access else fx.NO_CUBE_DECISION,
                     "checker_key": q["key"], "cube_key": ck})
    for q in by_key.values():
        if q["kind"] != "cube" or q["key"] in covered:
            continue
        d1, d2, post, half = fx.continuation_move(q["board"])
        recs.append({"board": q["board"], "cv": q["cv"], "owner": q["owner"], "dice": [d1, d2],
                     "post": post, "half_moves": half, "double": fx.NO_DOUBLE,
                     "checker_key": None, "cube_key": q["key"]})
    for r in recs:
        r["cube_code"] = fx.cube_code(r["cv"], r["owner"])
        if fx.apply_half_moves(r["board"], r["half_moves"]) != list(r["post"]):
            raise RuntimeError(f"half-moves do not replay the played move: {r}")
    return recs


def build_stream(tpl: dict, recs: list[dict]) -> bytes:
    stream = bytearray(fx.build_shard_stream(tpl, recs))
    stream[xf._HDR_JACOBY] = 1 if JACOBY else 0
    stream[xf._HDR_BEAVER] = 1 if BEAVER else 0
    return bytes(stream)


def verify(path: Path, recs: list[dict]) -> None:
    """``export_folder_benchmark_xg.verify_shard`` checks every record; it also
    insists on a Beaver header, which these shards deliberately lack, so the
    header is checked here and that one field patched for the call."""
    hdr = xf.parse_header(xf.XgArchive.load(path).get("temp.xg"), 0)
    if hdr["jacoby"] != JACOBY or hdr["beaver"] != BEAVER or hdr["match_length"] != 99999:
        raise AssertionError(f"{path.name}: unexpected match header {hdr}")
    orig = xf.parse_header
    xf.parse_header = lambda data, off=0: {**orig(data, off), "jacoby": True, "beaver": True}
    try:
        fx.verify_shard(path, recs)
    finally:
        xf.parse_header = orig


def find_template(arg: Path | None) -> Path:
    for p in ([arg] if arg else list(DEFAULT_TEMPLATES)):
        if p and p.exists():
            return p
    raise SystemExit(f"no XG template found (tried {arg or [str(p) for p in DEFAULT_TEMPLATES]});"
                     f" copy any XG-analysed .xg to {DEFAULT_TEMPLATES[0]}")


def write_round(run: Run, recs: list[dict], template: Path) -> Path:
    n = len(run.round_dirs()) + 1
    rd = run.dir / f"round_{n:03d}"
    rd.mkdir()
    tpl = _templates(template)
    for s in range(0, len(recs), PER_SHARD):
        chunk = recs[s:s + PER_SHARD]
        path = rd / f"shard_{s // PER_SHARD:03d}.xg"
        write_xg(template, build_stream(tpl, chunk), path)
        with open(str(path) + SIDECAR, "w", encoding="utf-8") as f:
            for gi, r in enumerate(chunk):
                f.write(json.dumps({"game_index": gi, "board": r["board"], "dice": r["dice"],
                                    "cv": r["cv"], "owner": r["owner"], "double": r["double"],
                                    "checker_key": r["checker_key"], "cube_key": r["cube_key"]},
                                   separators=(",", ":")) + "\n")
        verify(path, chunk)
    return rd


def unanswered_in_last_round(run: Run) -> tuple[int, int]:
    rds = run.round_dirs()
    if not rds:
        return 0, 0
    total = missing = 0
    for sc in rds[-1].glob("shard_*.xg" + SIDECAR):
        for row in _read_jsonl(sc):
            for k in (row.get("cube_key"), row.get("checker_key")):
                if k:
                    total += 1
                    missing += run.xg.get(k) is None
    return missing, total


def summarise(results: dict[str, dict]) -> dict:
    done = [r for r in results.values() if r["status"] == "done"]
    q = [q for r in results.values() for q in r["queries"]]
    return {"games": len(results), "done": len(done), "pending": len(results) - len(done),
            "queries": len(q), "unique_queries": len({x["key"] for x in q}),
            "checker_queries": sum(1 for x in q if x["kind"] == "checker"),
            "cube_queries": sum(1 for x in q if x["kind"] == "cube")}


def cmd_round(args) -> None:
    run = Run(args.dir)
    standin = run.cfg["opponent"].removeprefix("standin:") if run.cfg["opponent"] != "xg" else None
    if args.simulate_xg and standin is None:
        raise SystemExit("this run was initialised against XG; --simulate-xg needs a run made "
                         "with `init --standin LEVEL`")
    started = time.perf_counter()

    if standin:
        oracle = SageOracle(standin, run.dir / "standin_cache.jsonl")
        rounds = 0
        while True:
            results = run.play_all(args.lookahead)
            s = summarise(results)
            print(f"round {rounds + 1}: {s['done']}/{s['games']} games done, "
                  f"{s['unique_queries']} unanswered stand-in decisions "
                  f"({s['checker_queries']} checker / {s['cube_queries']} cube) "
                  f"[{time.perf_counter() - started:.0f}s, Sage computed {run.sage.computed}]",
                  flush=True)
            if not s["pending"]:
                break
            rounds += 1
            for q in {x["key"]: x for r in results.values() for x in r["queries"]}.values():
                if q["kind"] == "cube":
                    ans = oracle.cube(q["board"], q["cv"], q["owner"])
                else:
                    ans = oracle.checker(q["board"], q["cv"], q["owner"], *q["dice"])
                run.xg.add(q["key"], ans, {"round": f"sim_{rounds:03d}", "level": standin})
            _append_jsonl(run.dir / "rounds.jsonl", {"round": rounds, **s})
        print(f"all {s['games']} games finished after {rounds} answer round(s)")
        return

    h = harvest(run)
    if h["counts"]:
        print(f"harvested: {h['counts']}")
        print(f"XG levels on the analysed records: {h['levels']}")
    if run.xg.conflicts:
        print(f"WARNING: XG gave {len(run.xg.conflicts)} different answer(s) for a decision it had "
              f"already answered; first: {run.xg.conflicts[0]}")
    missing, total = unanswered_in_last_round(run)
    if missing and not args.force:
        raise SystemExit(f"{run.round_dirs()[-1].name}: {missing} of {total} exported decisions have "
                         f"no XG answer yet. Run XG's Batch Analyze on that folder ('Save games "
                         f"after analyze' ON), or pass --force to export the missing ones again.")

    results = run.play_all(args.lookahead)
    s = summarise(results)
    print(f"{s['done']}/{s['games']} games finished; {s['pending']} wait on "
          f"{s['unique_queries']} XG decisions [{time.perf_counter() - started:.0f}s, "
          f"Sage computed {run.sage.computed} decisions]")
    if not s["pending"]:
        print("All games finished. Next: python scripts/xg_h2h.py report --dir", run.dir)
        return
    recs = plan_records([q for r in results.values() for q in r["queries"]])
    rd = write_round(run, recs, find_template(args.template))
    n_cube = sum(1 for r in recs if r["cube_key"])
    n_chk = sum(1 for r in recs if r["checker_key"])
    _append_jsonl(run.dir / "rounds.jsonl", {"round": rd.name, **s, "records": len(recs),
                                             "cube_answers_asked": n_cube,
                                             "checker_answers_asked": n_chk})
    print(f"wrote {rd} : {len(recs)} one-decision games ({n_chk} checker plays, {n_cube} cube "
          f"decisions) in {math.ceil(len(recs) / PER_SHARD)} shard(s), verified")
    print(f"Next: XG -> File -> Batch Analyze on {rd} at the run's XG level, "
          f"'Save games after analyze' ON; then run `round` again.")


# --- report --------------------------------------------------------------------


def _mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def game_luck(run: Run, res: dict, sage_seat: int, analyzer, cache: dict, cache_path: Path) -> float:
    """Sage's net luck in points: its rolls' luck minus XG's, each times the
    cube value it was rolled at."""
    net = 0.0
    for e in res["events"]:
        if e["kind"] != "move":
            continue
        d1, d2 = e["dice"]
        opening = e["t"] == 0
        key = f"L|{_b(e['board'])}|{e['cv']}|{e['owner']}|{d1}{d2}|{int(opening)}"
        if key not in cache:
            lr = analyzer.roll_luck(e["board"], d1, d2, cube_value=e["cv"], cube_owner=e["owner"],
                                    jacoby=JACOBY, beaver=BEAVER, is_opening_roll=opening)
            if lr is None:
                raise RuntimeError(f"no luck for {key}")
            cache[key] = lr.luck
            _append_jsonl(cache_path, {"key": key, "luck": lr.luck})
        pts = cache[key] * e["cv"]
        net += pts if e["seat"] == sage_seat else -pts
    return net


def cmd_report(args) -> None:
    run = Run(args.dir)
    results = run.play_all()
    s = summarise(results)
    if s["pending"] and not args.partial:
        raise SystemExit(f"{s['pending']} of {s['games']} games are not finished; run more rounds "
                         f"(or --partial to report the finished ones, which biases the result)")
    seat = {gid: st for gid, _, st in run.games}
    done = {gid: r for gid, r in results.items() if r["status"] == "done"}

    luck = {}
    if args.luck:
        an = BgBotAnalyzer(eval_level="2ply")
        lpath = run.dir / "luck_cache.jsonl"
        lcache = {r["key"]: r["luck"] for r in _read_jsonl(lpath)}
        for gid, r in done.items():
            luck[gid] = game_luck(run, r, seat[gid], an, lcache, lpath)

    # agreement of Sage's prediction with XG, per decision type
    agree = Counter()
    for gid, r in done.items():
        for e in r["events"]:
            if e["src"] != "xg":
                continue
            b, cv, ow = e["board"], e["cv"], e["owner"]
            if e["kind"] == "move":
                agree["checker"] += 1
                agree["checker_same"] += run.sage.checker(b, cv, ow, *e["dice"])["board"] == e["post"]
            elif e["kind"] == "offer":
                agree["offer"] += 1
                agree["offer_same"] += run.sage.cube(b, cv, ow)["double"] == e["double"]
            elif e["kind"] == "response":
                agree["take"] += 1
                agree["take_same"] += run.sage.cube(b, cv, ow)["take"] == e["take"]

    per_game = [r["result"] for r in done.values()]
    pairs = []
    for p in range(run.cfg["pairs"]):
        a, b = done.get(f"p{p:04d}a"), done.get(f"p{p:04d}b")
        if a and b:
            pairs.append((a["result"] + b["result"]) / 2)
    print(f"Sage {run.cfg['sage_level']} vs {run.cfg['opponent']}: {len(done)}/{s['games']} games "
          f"finished, {len(run.round_dirs()) or len(_read_jsonl(run.dir / 'rounds.jsonl'))} "
          f"round(s); Jacoby {'on' if JACOBY else 'off'}, beaver {'on' if BEAVER else 'off'}")
    m, se = _mean_se(per_game)
    print(f"  points per game for Sage, per game:      {m:+.3f} +- {se:.3f}  (n={len(per_game)})")
    m, se = _mean_se(pairs)
    print(f"  points per game for Sage, duplicate pairs: {m:+.3f} +- {se:.3f}  (n={len(pairs)})")
    if luck:
        adj = [done[g]["result"] - luck[g] for g in done]
        m, se = _mean_se(adj)
        print(f"  luck-adjusted, per game:                 {m:+.3f} +- {se:.3f}")
        adj_pairs = []
        for p in range(run.cfg["pairs"]):
            ga, gb = f"p{p:04d}a", f"p{p:04d}b"
            if ga in done and gb in done:
                adj_pairs.append((done[ga]["result"] - luck[ga] + done[gb]["result"] - luck[gb]) / 2)
        m, se = _mean_se(adj_pairs)
        print(f"  luck-adjusted, duplicate pairs:          {m:+.3f} +- {se:.3f}")
    how = Counter((r["how"], r["points"]) for r in done.values())
    print(f"  endings (how, points): {dict(sorted(how.items()))}")
    turns = [r["turns"] for r in done.values()]
    print(f"  turns per game: mean {sum(turns) / max(1, len(turns)):.1f}, max {max(turns, default=0)}")
    for k in ("checker", "offer", "take"):
        if agree[k]:
            print(f"  Sage predicted XG's {k:7s}: {agree[k + '_same']}/{agree[k]} "
                  f"({agree[k + '_same'] / agree[k]:.1%})")
    rows = [{"game": gid, "result": r["result"], "how": r["how"], "points": r["points"],
             "turns": r["turns"], "luck": luck.get(gid), "events": r["events"]}
            for gid, r in done.items()]
    out = run.dir / "games.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"  per-game records -> {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="create a run directory")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--pairs", type=int, required=True, help="duplicate game pairs (2 games each)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sage-level", default="3ply")
    p.add_argument("--standin", default=None, help="a Sage level standing in for XG (testing)")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("round", help="read XG's answers, replay, write the next round")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--lookahead", type=int, default=0,
                   help="stop each game after this many unanswered XG decisions (0 = play to the end)")
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="export again what XG left unanalysed")
    p.add_argument("--simulate-xg", action="store_true",
                   help="answer with the run's stand-in level instead of XG (stand-in runs only)")
    p.set_defaults(func=cmd_round)
    p = sub.add_parser("report", help="score the finished games")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--luck", action="store_true", help="also report luck-adjusted results")
    p.add_argument("--partial", action="store_true")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    args.dir = args.dir if args.dir.is_absolute() else Path.cwd() / args.dir
    args.func(args)


if __name__ == "__main__":
    main()
