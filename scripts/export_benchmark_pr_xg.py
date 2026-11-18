#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Have eXtreme Gammon analyse Benchmark PR decisions, as an independent referee.

The Benchmark PR reference (``data/benchmark_pr/rollouts.jsonl``) is Stage 5's
own rollout, and only the candidates Stage 5's 2-ply filter kept were rolled
out, so it leans towards Stage 5 (CLAUDE.md, "Benchmark PR rollout sample").
This script hands the same decisions to XG, the strongest engine there is, and
reads XG's analysis back, so a Sage change can be checked against a referee
that is not Sage.

XG has no API, so every selected decision is written as a ONE-DECISION GAME in
a native ``.xg`` archive (the layout of ``export_folder_benchmark_xg.py``,
whose record builders and shard checks this script reuses). A human runs XG's
Batch Analyze over the shards; XG stores its ranked candidates, with
probabilities, in each move record; ``score`` reads them back.

Steps
-----
1. ``python scripts/export_benchmark_pr_xg.py generate`` (default ``--subset
   race``: the pure-race decisions of the scored sample) writes
   ``data/benchmark_pr_xg/<subset>/bench_shard_NNN.xg`` plus a
   ``.sidecar.jsonl`` per shard. The record templates come from an
   XG-written game file (``--template``; any ``.xg`` from XG's Batch Analyze
   of ``run_sage_vs_sage_games.py`` transcripts will do).
2. Copy the subset folder once per XG level (XG rewrites files in place), e.g.
   ``data/benchmark_pr_xg/race_xg4ply/``. In XG (CrossOver works): Analyze ->
   Batch Analyze -> Choose the shards -> level -> tick "Save Games after
   analyze" -> Start.
3. ``python scripts/export_benchmark_pr_xg.py score --xg-dir
   data/benchmark_pr_xg/race_xg4ply`` prints XG's own PR against the Stage 5
   reference (per subset: bearoff / outside the bearoff DB) and writes
   ``<xg-dir>/xg_reference.jsonl``: per decision, XG's stored moves with their
   cubeless equity (from XG's probabilities) and the level XG evaluated each
   at. That file is the XG reference Sage's picks are scored against.
4. ``python scripts/export_benchmark_pr_xg.py sage --xg-dir <same> [--gnubg]``
   scores the production model at 1-ply / 2-ply (bearoff DB on), with its own
   and with Stage 5's 200h pure-race net, plus XG's #1, the Stage 5
   reference's best move and optionally GNUbg, against BOTH references --
   the Stage 5 rollout and XG -- per subset, and writes the per-decision
   errors to ``<xg-dir>/sage_vs_xg.json``. ``--race-net "label=file:hidden"``
   adds further pure-race nets.

Encoding notes
--------------
* Benchmark PR is cubeless. Every game is a money game with the cube centred;
  the match header's Jacoby flag is switched OFF (the template carries the
  run_sage_vs_sage default, Jacoby + Beaver on), because with Jacoby a
  centred cube makes XG rank moves as if gammons did not count. XG's
  probabilities are cubeless either way; ``score`` computes the cubeless
  equity from them, so only XG's ranking (which moves it evaluates at full
  depth) is cubeful.
* The move record plays the reference's best candidate; XG analyses the
  position whatever was played.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
import export_folder_benchmark_xg as fx  # noqa: E402
from bgsage import possible_moves, xg_file as xf  # noqa: E402
from bgsage.weights import bearoff_db_path  # noqa: E402
from export_pasko_benchmark_xg import _templates, derive_half_moves, write_xg  # noqa: E402

BENCHMARK_DIR = _PROJECT_ROOT / "data" / "benchmark_pr"
DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "benchmark_pr_xg"
DEFAULT_TEMPLATE = DEFAULT_OUT_DIR / "template.xg"
SUBSETS = ("race", "contact", "all")
PR_MULTIPLIER = 500.0


def _rel(p: Path) -> Path:
    p = p.resolve()
    return p.relative_to(_PROJECT_ROOT) if p.is_relative_to(_PROJECT_ROOT) else p


def load_sample(decision_step: int) -> list[dict]:
    """The decisions ``score_benchmark_pr.py --decision-step N`` scores, with
    their index in decisions.json and the rollout equity of every candidate."""
    saved = json.load(open(BENCHMARK_DIR / "decisions.json"))
    eq = {}
    with open(BENCHMARK_DIR / "rollouts.jsonl") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                eq[tuple(rec["board"])] = rec["equity"]
    out = []
    for idx in range(0, len(saved["decisions"]), decision_step):
        d = saved["decisions"][idx]
        cands = d["candidates"]
        if len(cands) < 2 or not all(tuple(c) in eq for c in cands):
            continue                      # single legal move, or not rolled out
        out.append({"index": idx, "board": d["board"], "dice": d["dice"],
                    "candidates": cands, "rollout_equities": [eq[tuple(c)] for c in cands],
                    "n_total_candidates": d.get("n_total_candidates")})
    return out


def select(decisions: list[dict], subset: str, db) -> list[dict]:
    bgbot_cpp.init_escape_tables()
    for d in decisions:
        d["plan"] = bgbot_cpp.classify_game_plan(d["board"])
        d["bearoff"] = bool(db.is_bearoff(d["board"])) if db is not None else False
    if subset == "race":
        return [d for d in decisions if d["plan"] == "purerace"]
    if subset == "contact":
        return [d for d in decisions if d["plan"] != "purerace"]
    return decisions


def plan_games(decisions: list[dict]) -> tuple[list[dict], list[int]]:
    """One-decision games, and the indices of the decisions left out because
    some of their saved candidates are not legal moves. (23 decisions of the
    1-in-20 sample carry one, 17 of them as the reference's best move -- a
    move that leaves a die unplayed although it could be played -- so every
    engine is charged on them; they are not worth XG's time.)"""
    games, illegal = [], []
    for d in decisions:
        board = list(d["board"])
        d1, d2 = d["dice"]
        legal = possible_moves(board, d1, d2)
        if any(list(c) not in legal for c in d["candidates"]):
            illegal.append(d["index"])
            continue
        best = max(range(len(d["candidates"])), key=lambda i: d["rollout_equities"][i])
        post = list(d["candidates"][best])
        half = derive_half_moves(board, post, d1, d2)
        if fx.apply_half_moves(board, half) != post:
            raise ValueError(f"decision {d['index']}: half-moves do not replay the move")
        games.append({"key": d["index"], "folder": "benchmark_pr", "kind": "checker",
                      "board": board, "cube_value": 1, "cube_owner": "centered",
                      "cube_code": 0, "double": fx.NO_CUBE_DECISION, "dice": [d1, d2],
                      "post": post, "half_moves": half, "plan": d["plan"],
                      "bearoff": d["bearoff"], "candidates": d["candidates"],
                      "rollout_equities": d["rollout_equities"],
                      "n_total_candidates": d["n_total_candidates"]})
    return games, illegal


def build_stream(tpl: dict, games: list[dict]) -> bytes:
    stream = bytearray(fx.build_shard_stream(tpl, games))
    stream[xf._HDR_JACOBY] = 0            # cubeless benchmark: gammons must count
    return bytes(stream)


def verify(path: Path, games: list[dict]) -> None:
    """``export_folder_benchmark_xg.verify_shard`` checks every record; it also
    insists on a Jacoby header, which these shards deliberately lack, so that
    one field is checked here instead."""
    hdr = xf.parse_header(xf.XgArchive.load(path).get("temp.xg"), 0)
    if hdr["jacoby"] or not hdr["beaver"] or hdr["match_length"] != 99999:
        raise AssertionError(f"{path.name}: unexpected match header {hdr}")
    orig = xf.parse_header
    xf.parse_header = lambda data, off=0: {**orig(data, off), "jacoby": True}
    try:
        fx.verify_shard(path, games)
    finally:
        xf.parse_header = orig


def sidecar_row(gi: int, g: dict) -> dict:
    return {"game_index": gi, "key": g["key"], "board": g["board"], "dice": g["dice"],
            "plan": g["plan"], "bearoff": g["bearoff"], "played_board": g["post"],
            "candidates": g["candidates"], "rollout_equities": g["rollout_equities"],
            "n_total_candidates": g["n_total_candidates"]}


def cmd_generate(args) -> None:
    if not args.template.exists():
        raise SystemExit(f"template .xg not found: {args.template} (copy any XG-analysed "
                         f"seed_N.xg there, or pass --template)")
    tpl = _templates(args.template)
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path() or ""):
        raise SystemExit("bearoff database not found (scripts/generate_bearoff_db.py)")
    decisions = select(load_sample(args.decision_step), args.subset, db)
    if args.limit:
        decisions = decisions[:args.limit]
    games, illegal = plan_games(decisions)
    out_dir = args.out_dir / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("bench_shard_*.xg")) + \
            list(out_dir.glob("bench_shard_*" + fx.SIDECAR_SUFFIX)):
        stale.unlink()
    t0 = time.perf_counter()
    n_shards = (len(games) + args.per_file - 1) // args.per_file
    for s in range(n_shards):
        chunk = games[s * args.per_file:(s + 1) * args.per_file]
        path = out_dir / fx.SHARD_NAME.format(s)
        write_xg(args.template, build_stream(tpl, chunk), path)
        with open(str(path) + fx.SIDECAR_SUFFIX, "w") as f:
            for gi, g in enumerate(chunk):
                f.write(json.dumps(sidecar_row(gi, g), separators=(",", ":")) + "\n")
        verify(path, chunk)
    by = Counter(("bearoff" if g["bearoff"] else "outside DB") if g["plan"] == "purerace"
                 else g["plan"] for g in games)
    print(f"{len(games)} decisions ({dict(by)}) -> {n_shards} shard(s) in "
          f"{_rel(out_dir)}, verified [{time.perf_counter() - t0:.1f}s]")
    if illegal:
        print(f"left out {len(illegal)} decision(s) with an illegal saved candidate: {illegal}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def cubeless_equity(probs) -> float:
    w, gw, bw, gl, bl = probs
    return 2 * w - 1 + gw - gl + bw - bl


def harvest(xg_dir: Path, gen_dir: Path) -> list[dict]:
    out = []
    for shard in sorted(xg_dir.glob("bench_shard_*.xg")):
        side = Path(str(shard) + fx.SIDECAR_SUFFIX)
        if not side.exists():
            side = gen_dir / side.name
        rows = [json.loads(line) for line in open(side) if line.strip()]
        tx = xf.XgArchive.load(shard).get("temp.xg")
        games = fx.iter_games(tx)
        if len(games) != len(rows):
            raise SystemExit(f"{shard.name}: {len(games)} games, sidecar has {len(rows)}")
        for row, g in zip(rows, games):
            rec = xf.parse_move_record(tx, g["move"][0])
            if list(rec["mover_board"]) != row["board"] or rec["dice"] != row["dice"]:
                raise SystemExit(f"{shard.name} game {row['game_index']}: record does not "
                                 f"match its sidecar row")
            moves = [{"board": m["board"], "level": xf.player_level_label(m["level"]),
                      "probs": xf.xg_eval_to_probs(m["eval"]),
                      "equity_xg": float(m["eval"][6])} for m in rec["moves"]]
            for m in moves:
                m["equity"] = cubeless_equity(m["probs"])
            out.append({**row, "analyze_m": xf.player_level_label(rec["analyze_m"]),
                        "xg_moves": moves})
    return out


def cmd_score(args) -> None:
    gen_dir = args.out_dir / args.subset
    rows = harvest(args.xg_dir, gen_dir)
    analysed = [r for r in rows if r["xg_moves"]]
    print(f"{len(rows)} decisions, {len(analysed)} analysed by XG "
          f"(levels: {dict(Counter(r['analyze_m'] for r in analysed))})")
    groups: dict[str, list[float]] = {}
    outside = Counter()
    for r in analysed:
        ref = {tuple(c): e for c, e in zip(r["candidates"], r["rollout_equities"])}
        pick = tuple(r["xg_moves"][0]["board"])
        best = max(ref.values())
        if pick in ref:
            err = best - ref[pick]
        else:                                 # outside the rolled-out set: charge the worst
            err = best - min(ref.values())
        name = ("bearoff" if r["bearoff"] else "outside DB") if r["plan"] == "purerace" \
            else r["plan"]
        for k in (name, "all"):
            groups.setdefault(k, []).append(err)
            outside[k] += pick not in ref
    print("XG's #1 move against the Stage 5 rollout reference:")
    print(f"  {'subset':12}{'n':>6}{'PR':>8}{'picks outside ref':>20}")
    for k, errs in sorted(groups.items(), key=lambda kv: kv[0] != "all"):
        print(f"  {k:12}{len(errs):>6}{sum(errs) / len(errs) * PR_MULTIPLIER:>8.2f}"
              f"{outside[k]:>20}")
    out = args.xg_dir / "xg_reference.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {_rel(out)}")


# ---------------------------------------------------------------------------
# sage: Sage (and GNUbg) scored against the XG reference
# ---------------------------------------------------------------------------

#: Pure-race nets to score in place of the model's own (``race_net_swap.py``).
RACE_NETS = {"own": None, "s5 200h": ("sl_s5_purerace.weights.best", 200)}
REF_KEYS = ("board", "dice", "candidates", "rollout_equities")


def xg_decision(r: dict) -> dict | None:
    """The decision with XG's stored moves as candidates, each valued at the
    cubeless equity of XG's own evaluation of it. That mixes depths: XG takes
    only its leading moves to the full level (``analyze_m``) and leaves the
    rest at the ply its filter dropped them at -- the convention XG's own error
    figures use. A pick XG did not store is charged the worst stored move, as
    the Benchmark PR scorers do with picks outside the rolled-out set."""
    moves = r.get("xg_moves") or []
    if len(moves) < 2:
        return None
    return {"board": r["board"], "dice": r["dice"],
            "candidates": [m["board"] for m in moves],
            "rollout_equities": [m["equity"] for m in moves]}


def subset_name(r: dict) -> str:
    if r["plan"] != "purerace":
        return r["plan"]
    return "bearoff" if r["bearoff"] else "outside DB"


def _gnubg_err(args):
    import score_benchmark_pr_gnubg as sg
    dec, plies = args
    res = sg.score_decision_gnubg(dec, plies)
    return None if res is None else res[1]


def cmd_sage(args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from bgsage.weights import PRODUCTION_MODEL, WeightConfig

    rows = [json.loads(line) for line in open(args.xg_dir / "xg_reference.jsonl")]
    rows = [r for r in rows if xg_decision(r) is not None]
    refs = {"stage5 RO": [{k: r[k] for k in REF_KEYS} for r in rows],
            "XG": [xg_decision(r) for r in rows]}
    names = [subset_name(r) for r in rows]
    groups = {"all": list(range(len(rows)))}
    for i, n in enumerate(names):
        groups.setdefault(n, []).append(i)
    print(f"{len(rows)} decisions analysed by XG at "
          f"{dict(Counter(r['analyze_m'] for r in rows))}; "
          f"subsets {({k: len(v) for k, v in groups.items()})}")

    # errors[config][ref] = per-decision error (equity), or None when unscored
    errors: dict[str, dict[str, list]] = {}

    def by_pick(label, pick_of):
        errors[label] = {}
        for ref, decs in refs.items():
            errs = []
            for r, d in zip(rows, decs):
                eq = dict(zip(map(tuple, d["candidates"]), d["rollout_equities"]))
                p = pick_of(r)
                errs.append(max(eq.values()) - eq.get(tuple(p), min(eq.values())))
            errors[label][ref] = errs

    by_pick("XG #1", lambda r: r["xg_moves"][0]["board"])
    by_pick("stage5 RO best", lambda r: r["candidates"][max(
        range(len(r["candidates"])), key=lambda i: r["rollout_equities"][i])])

    bgbot_cpp.init_escape_tables()
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path() or ""):
        raise SystemExit("bearoff database not found")
    model = args.model or PRODUCTION_MODEL
    w = WeightConfig.from_model(model)
    nets = dict(RACE_NETS)
    for spec in args.race_net:
        label, rest = spec.split("=", 1)
        fname, nh = rest.rsplit(":", 1)
        nets[label] = (fname, int(nh))
    for net_label, net in nets.items():
        paths, hidden = list(w.weight_paths_list), list(w.hidden_sizes_list)
        if net is not None:
            path = Path(net[0]) if Path(net[0]).exists() else _PROJECT_ROOT / "models" / net[0]
            if not path.exists():
                raise SystemExit(f"race net not found: {net[0]}")
            paths[0], hidden[0] = str(path), net[1]
        base = bgbot_cpp.BearoffStrategy(
            bgbot_cpp.create_strategy(w.strategy_type, paths, hidden), db)
        m2 = bgbot_cpp.create_multipy(w.strategy_type, paths, hidden, n_plies=2)
        bgbot_cpp.multipy_set_bearoff_db(m2, db)
        for plies, strat, filt in ((1, base, None), (2, m2, base)):
            label = f"{model} {plies}P" + ("" if net is None else f" +{net_label} race")
            errors[label] = {}
            for ref, decs in refs.items():
                errs = []
                for d in decs:
                    res = bgbot_cpp.score_benchmark_pr_strategy([d], strat, filt, n_threads=1)
                    errs.append(res["total_error"] if res["total_decisions"] else None)
                errors[label][ref] = errs
            print(f"  {label} done", flush=True)

    if args.gnubg:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for plies in (1, 2):
                label = f"GNUbg {plies}P"
                errors[label] = {ref: list(pool.map(_gnubg_err, [(d, plies) for d in decs],
                                                    chunksize=8))
                                 for ref, decs in refs.items()}
                print(f"  {label} done", flush=True)

    cols = [g for g in ("all", "outside DB", "bearoff") if g in groups] + \
        [g for g in groups if g not in ("all", "outside DB", "bearoff")]
    for ref in refs:
        print(f"\nPR against the {ref} reference")
        print(f"  {'player':26}" + "".join(f"{c:>13}" for c in cols))
        for label, per_ref in errors.items():
            line = f"  {label:26}"
            for c in cols:
                v = [per_ref[ref][i] for i in groups[c] if per_ref[ref][i] is not None]
                line += f"{(sum(v) / len(v) * PR_MULTIPLIER if v else float('nan')):>13.2f}"
            print(line)
    out = args.xg_dir / "sage_vs_xg.json"
    out.write_text(json.dumps({"keys": [r["key"] for r in rows], "subsets": names,
                               "errors": errors}))
    print(f"\nper-decision errors -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the one-decision .xg shards")
    g.add_argument("--subset", choices=SUBSETS, default="race")
    g.add_argument("--decision-step", type=int, default=20)
    g.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    g.add_argument("--per-file", type=int, default=200)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    s = sub.add_parser("score", help="read XG's analysis back")
    s.add_argument("--xg-dir", type=Path, required=True)
    s.add_argument("--subset", choices=SUBSETS, default="race")
    s.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    v = sub.add_parser("sage", help="score Sage (and GNUbg) against XG's analysis "
                                    "and against the Stage 5 rollout reference")
    v.add_argument("--xg-dir", type=Path, required=True)
    v.add_argument("--model", default=None, help="default: the production model")
    v.add_argument("--gnubg", action="store_true", help="also score GNUbg 1-ply / 2-ply")
    v.add_argument("--race-net", action="append", default=[],
                   help='extra pure-race net "label=file:hidden" (file under models/ or a '
                        'path; repeatable), scored in place of the model\'s own')
    v.add_argument("--workers", type=int, default=12, help="parallel GNUbg processes")
    args = ap.parse_args()
    {"generate": cmd_generate, "score": cmd_score, "sage": cmd_sage}[args.cmd](args)


if __name__ == "__main__":
    main()
