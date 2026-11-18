#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""XG as an independent referee for the money benchmark's cube decisions.

The money benchmark's cube references (``data/money_benchmark/benchmark.json``)
are Sage's own analyses: 3-ply, 3T or full rollouts in which Sage makes the
in-trial cube decisions itself. A cube-life change tuned against them could
partly be learning to agree with Sage. This script lets eXtreme Gammon grade
the same decisions instead.

1. ``generate`` writes every scoreable cube position as a one-decision game in
   native ``.xg`` shards (``data/money_cube_xg/bench_shard_NNN.xg`` plus a
   ``.sidecar.jsonl`` each), with the record layout of
   ``export_folder_benchmark_xg.py``: a cube record with "no double" played
   (XG analyses it and stores ND / DT / DP and its recommendation), then a
   filler roll. The template's match header carries Jacoby and Beaver, as the
   benchmark does.
2. In XG (it runs under CrossOver on the Mac): File -> Batch Analyze on the
   folder, level "World Class", "Save games after analyze" ON. XG writes its
   analysis into each file in place.
3. ``score --xg-dir DIR`` reads XG's equities back into ``DIR/xg_reference.jsonl``
   and prints XG's own cube PR against the Sage reference, as a sanity check.
4. ``sage --xg-dir DIR`` plays every decision with Sage at 1-, 2- and 3-ply,
   with the engine-default cube life ("old", ``MONEY_CONTACT_CUBE_LIFE``
   emptied) and the current one ("new"), and scores both against the Sage
   reference and against XG, with paired new-minus-old differences.

Scoring is ``benchmark_money._score_cube``'s: the doubler's error
``max(0, max(ND, min(DT, DP)) - actual)`` and the receiver's
``max(0, actual - min(DT, DP))``, over the sub-decisions the benchmark marks
live (``has_double`` / ``has_take``), with the reference's equities swapped
for XG's in the XG column. PR = mean error x 500.

Usage::

    python scripts/export_money_cube_xg.py generate
    python scripts/export_money_cube_xg.py score --xg-dir data/money_cube_xg
    python scripts/export_money_cube_xg.py sage --xg-dir data/money_cube_xg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import benchmark_money as bm  # noqa: E402
import export_folder_benchmark_xg as fx  # noqa: E402
from bgsage import xg_file as xf  # noqa: E402
from export_pasko_benchmark_xg import _templates, write_xg  # noqa: E402

DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "money_cube_xg"
DEFAULT_TEMPLATE = DEFAULT_OUT_DIR / "template.xg"
PR_MULTIPLIER = 500.0
LEVELS = ("1ply", "2ply", "3ply")


def _rel(p: Path) -> Path:
    p = p.resolve()
    return p.relative_to(_PROJECT_ROOT) if p.is_relative_to(_PROJECT_ROOT) else p


def load_cube_entries() -> list[dict]:
    """The cube positions ``benchmark_money.benchmark_pr`` scores, in data-set order."""
    data = bm._read_dataset(bm.DEFAULT_DATASET)
    return [e for e in data["decisions"]
            if e["kind"] == "cube" and bm._missing_tier(e) is None
            and (e.get("has_double") or e.get("has_take"))]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def plan_games(entries: list[dict]) -> tuple[list[dict], Counter]:
    games, skipped = [], Counter()
    for e in entries:
        if e["cube_owner"] == "opponent":
            skipped["opponent_cube"] += 1     # the mover has no decision
            continue
        board = list(e["board"])
        d1, d2, post, half = fx.continuation_move(board)
        if fx.apply_half_moves(board, half) != post:
            raise ValueError(f"{e['key']}: filler half-moves do not replay")
        games.append({"key": e["key"], "folder": "money_cube", "kind": "cube", "board": board,
                      "cube_value": e["cube_value"], "cube_owner": e["cube_owner"],
                      "cube_code": fx.cube_code(e["cube_value"], e["cube_owner"]),
                      "double": fx.NO_DOUBLE, "dice": [d1, d2], "post": post,
                      "half_moves": half, "has_double": bool(e.get("has_double")),
                      "has_take": bool(e.get("has_take"))})
    return games, skipped


def cmd_generate(args) -> None:
    if not args.template.exists():
        raise SystemExit(f"template .xg not found: {args.template} (copy any XG-analysed "
                         f"Sage-vs-Sage seed_N.xg there -- its header must carry Jacoby "
                         f"and Beaver -- or pass --template)")
    tpl = _templates(args.template)
    entries = load_cube_entries()
    if args.limit:
        entries = entries[:args.limit]
    games, skipped = plan_games(entries)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(args.out_dir.glob("bench_shard_*.xg")) + \
            list(args.out_dir.glob("bench_shard_*" + fx.SIDECAR_SUFFIX)):
        stale.unlink()
    t0 = time.perf_counter()
    n_shards = (len(games) + args.per_file - 1) // args.per_file
    for s in range(n_shards):
        chunk = games[s * args.per_file:(s + 1) * args.per_file]
        path = args.out_dir / fx.SHARD_NAME.format(s)
        write_xg(args.template, fx.build_shard_stream(tpl, chunk), path)
        with open(str(path) + fx.SIDECAR_SUFFIX, "w") as f:
            for gi, g in enumerate(chunk):
                f.write(json.dumps(fx.sidecar_row(gi, g), separators=(",", ":")) + "\n")
        fx.verify_shard(path, chunk)        # also checks the Jacoby/Beaver header
    owners = Counter((g["cube_owner"], g["cube_value"]) for g in games)
    print(f"{len(games)} cube positions ({sum(g['has_double'] for g in games)} doubler + "
          f"{sum(g['has_take'] for g in games)} receiver decisions; cube states "
          f"{dict(owners)}) -> {n_shards} shard(s) in {_rel(args.out_dir)}, verified "
          f"[{time.perf_counter() - t0:.1f}s]")
    if skipped:
        print(f"skipped: {dict(skipped)}")


# ---------------------------------------------------------------------------
# score: read XG's analysis back
# ---------------------------------------------------------------------------


def harvest(xg_dir: Path, gen_dir: Path) -> tuple[list[dict], int]:
    rows, unanalysed = [], 0
    for shard in sorted(xg_dir.glob("bench_shard_*.xg")):
        side = Path(str(shard) + fx.SIDECAR_SUFFIX)
        if not side.exists():
            side = gen_dir / side.name
        sc = [json.loads(line) for line in open(side) if line.strip()]
        tx = xf.XgArchive.load(shard).get("temp.xg")
        games = fx.iter_games(tx)
        if len(games) != len(sc):
            raise SystemExit(f"{shard.name}: {len(games)} games, sidecar has {len(sc)}")
        for row, g in zip(sc, games):
            pick, rec = fx.extract_cube_pick(tx, g["cube"][0])
            if list(rec["mover_board"]) != row["board"]:
                raise SystemExit(f"{shard.name} game {row['game_index']}: cube record does "
                                 f"not match its sidecar row")
            if pick is None:
                unanalysed += 1
                continue
            rows.append({"key": row["key"], "board": row["board"],
                         "cube_value": row["cube_value"], "cube_owner": row["cube_owner"],
                         "has_double": row["has_double"], "has_take": row["has_take"],
                         "xg": {"nd": rec["equity_nd"], "dt": rec["equity_dt"],
                                "dp": rec["equity_dp"], "flag_double": rec["flag_double"],
                                "is_beaver": rec["is_beaver"],
                                "level": xf.player_level_label(rec["level"]),
                                **pick}})
    return rows, unanalysed


def errors(ref: dict, has_double: bool, has_take: bool, should_double: bool,
           should_take: bool) -> list[float]:
    """``benchmark_money._score_cube``'s sub-decision errors against ``ref``
    (keys nd / dt / dp)."""
    nd, dt, dp = ref["nd"], ref["dt"], ref["dp"]
    out = []
    if has_double:
        out.append(max(0.0, max(nd, min(dt, dp)) - (min(dt, dp) if should_double else nd)))
    if has_take:
        out.append(max(0.0, (dt if should_take else dp) - min(dt, dp)))
    return out


def sage_ref(entry: dict) -> dict:
    return {"nd": entry["equity_nd"], "dt": entry["equity_dt"], "dp": entry["equity_dp"]}


def pr(errs: list[float]) -> float:
    return sum(errs) / len(errs) * PR_MULTIPLIER if errs else float("nan")


def cmd_score(args) -> None:
    rows, unanalysed = harvest(args.xg_dir, args.out_dir)
    out = args.xg_dir / "xg_reference.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    by_key = {e["key"]: e for e in load_cube_entries()}
    dp = Counter(round(r["xg"]["dp"], 3) for r in rows)
    print(f"{len(rows)} positions analysed by XG, {unanalysed} unanalysed "
          f"(levels: {dict(Counter(r['xg']['level'] for r in rows))}; DP values {dict(dp)})")
    errs = []
    for r in rows:
        errs += errors(sage_ref(by_key[r["key"]]), r["has_double"], r["has_take"],
                       r["xg"]["should_double"], r["xg"]["should_take"])
    print(f"XG's own cube PR against the Sage reference: {pr(errs):.2f} over {len(errs)} "
          f"decisions -> {_rel(out)}")


# ---------------------------------------------------------------------------
# sage: score Sage old/new against both references
# ---------------------------------------------------------------------------


def sage_picks(rows: list[dict], level: str, policy: str, threads: int) -> list[tuple[bool, bool]]:
    import bgsage.analyzer as A
    from bgsage import BgBotAnalyzer
    saved = dict(A.MONEY_CONTACT_CUBE_LIFE)
    if policy == "old":
        A.MONEY_CONTACT_CUBE_LIFE.clear()
    try:
        an = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=1)

        def one(r):
            a = an.cube_action(r["board"], cube_value=r["cube_value"], cube_owner=r["cube_owner"],
                               jacoby=True, beaver=True)
            return bool(a.should_double), bool(a.should_take)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            return list(ex.map(one, rows))
    finally:
        A.MONEY_CONTACT_CUBE_LIFE.clear()
        A.MONEY_CONTACT_CUBE_LIFE.update(saved)


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    return mean * PR_MULTIPLIER, math.sqrt(var / n) * PR_MULTIPLIER


def cmd_sage(args) -> None:
    rows = [json.loads(line) for line in open(args.xg_dir / "xg_reference.jsonl") if line.strip()]
    by_key = {e["key"]: e for e in load_cube_entries()}
    refs = {"sage": [sage_ref(by_key[r["key"]]) for r in rows],
            "xg": [r["xg"] for r in rows]}

    def per_position(picks, ref_name):
        # one summed error per position, so pairing works across sub-decisions
        out, flat = [], []
        for r, ref, (sd, st) in zip(rows, refs[ref_name], picks):
            e = errors(ref, r["has_double"], r["has_take"], sd, st)
            out.append(sum(e))
            flat += e
        return out, flat

    n_dec = sum(r["has_double"] + r["has_take"] for r in rows)
    print(f"{len(rows)} positions, {n_dec} cube decisions; cube PR (blunders > "
          f"{bm.BLUNDER_THRESHOLD}) against each reference\n")
    print(f"{'player':22s} {'vs Sage reference':>22s} {'vs XG':>22s}")
    report = {"n_positions": len(rows), "n_decisions": n_dec, "rows": []}
    xg_picks = [(r["xg"]["should_double"], r["xg"]["should_take"]) for r in rows]
    for name, picks in [("XG World Class", xg_picks)]:
        cells = []
        for ref in ("sage", "xg"):
            _, flat = per_position(picks, ref)
            cells.append(f"{pr(flat):6.2f} ({sum(e > bm.BLUNDER_THRESHOLD for e in flat):3d})")
        print(f"{name:22s} {cells[0]:>22s} {cells[1]:>22s}")
    for level in args.levels:
        pp = {}
        for policy in ("old", "new"):
            t0 = time.perf_counter()
            picks = sage_picks(rows, level, policy, args.threads)
            cells, row = [], {"level": level, "policy": policy}
            for ref in ("sage", "xg"):
                pos, flat = per_position(picks, ref)
                pp[(policy, ref)] = pos
                blunders = sum(e > bm.BLUNDER_THRESHOLD for e in flat)
                cells.append(f"{pr(flat):6.2f} ({blunders:3d})")
                row[ref] = {"pr": pr(flat), "blunders": blunders}
            report["rows"].append(row)
            print(f"{'Sage ' + level + ' ' + policy:22s} {cells[0]:>22s} {cells[1]:>22s}"
                  f"   [{time.perf_counter() - t0:.0f}s]", flush=True)
        diffs = []
        for ref in ("sage", "xg"):
            m, se = paired(pp[("old", ref)], pp[("new", ref)])
            # paired per position, scaled to per-decision PR units
            scale = len(rows) / n_dec
            diffs.append(f"{m * scale:+6.2f} +- {se * scale:4.2f}")
            report["rows"][-1][f"paired_{ref}"] = [m * scale, se * scale]
        print(f"{'  new - old':22s} {diffs[0]:>22s} {diffs[1]:>22s}")
    out = args.xg_dir / "sage_vs_xg.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\n-> {_rel(out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    g.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    g.add_argument("--per-file", type=int, default=200)
    g.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("score")
    s.add_argument("--xg-dir", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p = sub.add_parser("sage")
    p.add_argument("--xg-dir", type=Path, required=True)
    p.add_argument("--levels", nargs="+", default=list(LEVELS))
    p.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    {"generate": cmd_generate, "score": cmd_score, "sage": cmd_sage}[args.cmd](args)


if __name__ == "__main__":
    main()
