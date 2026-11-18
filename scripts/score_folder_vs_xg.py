#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Score Sage's picks on a folder benchmark against XG AND against the
folder's rollout reference, side by side, on the same decisions.

Why: on the snake folder the two rollout references disagree about Sage's
levels -- stage11 3-ply is the worst level under the S9-play reference and the
best under the S11-play twin (CLAUDE.md, "Stage 11s"). XG is an independent
referee for both decision sets.

Workflow
--------
1. ``export_folder_benchmark_xg.py generate --folders snake "snake s11play"``
   writes the one-decision ``.xg`` shards (``data/backgame_xg/<slug>/``).
2. ``score_folder_vs_xg.py picks --folders snake "snake s11play"`` plays every
   exported decision with Sage at 1/2/3-ply and caches the picks in
   ``data/backgame_xg_scores/sage_<model>_<level>_<slug>_picks.jsonl``
   (resumable: decisions already in the file are skipped). Independent of XG,
   so it can run while XG works.
3. A copy of ``data/backgame_xg/`` is Batch-Analyzed in XG ("Save games after
   analyze" ON) at the chosen level.
4. ``score_folder_vs_xg.py score --xg-dir <copy>`` grades every cached Sage
   level and XG's own #1 against both references and prints PRs plus the
   paired level differences with their standard errors.

How XG is used as a reference
-----------------------------
* Checker: XG's stored candidates, each valued at XG's CUBEFUL equity
  (``eval[6]``; the folder decisions are money games with Jacoby + Beaver,
  as is the template header, and the rollout references are cubeful too).
  Depths are mixed -- XG takes only its leading moves to the full level and
  leaves the rest at the depth its filter dropped them -- the convention of
  XG's own error figures and of the Benchmark PR XG scorer. A pick XG did not
  store is charged the worst stored move and counted ("outside").
* Cube: XG's ND/DT/DP. Which sub-decisions are live is decided from XG's own
  values with ``score_backgame_pr.cube_subdecisions`` (the double is live
  unless trivial; the take only when XG says double and DT/DP are not
  trivially apart), so each reference grades its own live sub-decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import export_folder_benchmark_xg as fx  # noqa: E402
from benchmark_money import BLUNDER_THRESHOLD, PR_MULTIPLIER  # noqa: E402
from bgsage import xg_file as xf  # noqa: E402
from score_backgame_pr import cube_subdecisions, load_reference  # noqa: E402

DEFAULT_LEVELS = ("1ply", "2ply", "3ply")
DEFAULT_SCORES_DIR = fx.DEFAULT_SCORES_DIR


def picks_path(scores_dir: Path, model: str, level: str, folder: str) -> Path:
    return scores_dir / f"sage_{model}_{level}_{fx.folder_slug(folder)}_picks.jsonl"


def exported_entries(folder: str) -> list[dict]:
    """The reference rows the exporter writes as games (same filter)."""
    games, _ = fx.plan_folder(folder, load_reference(folder, None))
    keys = {g["key"] for g in games}
    return [e for e in load_reference(folder, None) if e["key"] in keys]


# ---------------------------------------------------------------------------
# picks
# ---------------------------------------------------------------------------


def cmd_picks(args) -> None:
    from bgsage import BgBotAnalyzer
    from bgsage.weights import PRODUCTION_MODEL, WeightConfigPair

    model = args.model or PRODUCTION_MODEL
    weights = WeightConfigPair.from_model(model)
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        analyzer = BgBotAnalyzer(weights=weights, eval_level=level, cubeful=True,
                                 parallel_threads=args.threads)
        for folder in args.folders:
            entries = exported_entries(folder)
            path = picks_path(args.scores_dir, model, level, folder)
            done = set()
            if path.exists():
                done = {json.loads(l)["key"] for l in path.read_text().splitlines() if l.strip()}
            todo = [e for e in entries if e["key"] not in done]
            print(f"{model} {level} {folder}: {len(entries)} decisions, {len(todo)} to play",
                  flush=True)
            started = time.perf_counter()
            with open(path, "a", encoding="utf-8") as f:
                for i, e in enumerate(todo, start=1):
                    if e["kind"] == "checker":
                        d1, d2 = e["dice"]
                        res = analyzer.checker_play(e["board"], d1, d2,
                                                    cube_value=e["cube_value"],
                                                    cube_owner=e["cube_owner"],
                                                    jacoby=True, beaver=True)
                        pick = list(res.moves[0].board) if res.moves else None
                    else:
                        a = analyzer.cube_action(e["board"], cube_value=e["cube_value"],
                                                 cube_owner=e["cube_owner"],
                                                 jacoby=True, beaver=True)
                        pick = {"should_double": bool(a.should_double),
                                "should_take": bool(a.should_take)}
                    f.write(json.dumps({"key": e["key"], "kind": e["kind"], "pick": pick},
                                       separators=(",", ":")) + "\n")
                    f.flush()
                    if i % 50 == 0:
                        el = time.perf_counter() - started
                        print(f"  {i}/{len(todo)} ({el:.0f}s, "
                              f"~{el / i * (len(todo) - i) / 60:.0f} min left)", flush=True)
            print(f"  done in {time.perf_counter() - started:.0f}s -> {path.name}", flush=True)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def harvest_xg(xg_dir: Path, gen_dir: Path, folder: str) -> dict[str, dict]:
    """key -> XG reference: ``{"kind": "checker", "moves": [(board, eq, level)],
    "level"}`` or ``{"kind": "cube", "nd", "dt", "dp", "flag_double", "level"}``.
    Unanalysed records are left out."""
    slug = fx.folder_slug(folder)
    files = sorted((xg_dir / slug).glob("bench_shard_*.xg")) or \
        sorted(xg_dir.glob(f"{slug}__bench_shard_*.xg"))
    out: dict[str, dict] = {}
    for path in files:
        sidecar = fx.load_sidecar(path, gen_dir / slug)
        tx = xf.XgArchive.load(path).get("temp.xg")
        games = fx.iter_games(tx)
        if len(games) != len(sidecar):
            raise ValueError(f"{path}: {len(games)} games, {len(sidecar)} sidecar rows")
        for sc, g in zip(sidecar, games):
            if sc["kind"] == "checker":
                rec = xf.parse_move_record(tx, g["move"][0])
                if list(rec["mover_board"]) != list(sc["board"]):
                    raise ValueError(f"{path} game {sc['game_index']}: board mismatch")
                if rec["n_moves"] <= 0 or not rec["moves"]:
                    continue
                out[sc["key"]] = {
                    "kind": "checker", "level": xf.player_level_label(rec["analyze_m"]),
                    "moves": [(tuple(m["board"]), float(m["eval"][6]),
                               xf.player_level_label(m["level"])) for m in rec["moves"]]}
            else:
                rec = xf.parse_cube_record(tx, g["cube"][0])
                if list(rec["mover_board"]) != list(sc["board"]):
                    raise ValueError(f"{path} game {sc['game_index']}: board mismatch")
                if rec["analyze_c"] < 0 or rec["flag_double"] == fx.UNANALYSED_FLAG_DOUBLE:
                    continue
                out[sc["key"]] = {"kind": "cube", "level": xf.player_level_label(rec["level"]),
                                  "nd": float(rec["equity_nd"]), "dt": float(rec["equity_dt"]),
                                  "dp": float(rec["equity_dp"]),
                                  "flag_double": int(rec["flag_double"])}
    return out


def cube_errors(nd, dt, dp, has_double, has_take, pick) -> list[float]:
    errs = []
    if has_double:
        actual = min(dt, dp) if pick["should_double"] else nd
        errs.append(max(0.0, max(nd, min(dt, dp)) - actual))
    if has_take:
        actual = dt if pick["should_take"] else dp
        errs.append(max(0.0, actual - min(dt, dp)))
    return errs


def errors_vs_rollout(entry: dict, pick) -> tuple[list[float], bool]:
    """Per-sub-decision errors against the folder's rollout reference."""
    if entry["kind"] == "checker":
        best = entry["moves"][0]["equity"]
        m = next((m for m in entry["moves"] if list(m["board"]) == list(pick)), None)
        if m is None:
            rolled = [x["equity"] for x in entry["moves"] if x["eval_level"] == "Rollout"]
            return [max(0.0, best - min(rolled or [x["equity"] for x in entry["moves"]]))], True
        return [max(0.0, best - m["equity"])], False
    has_double, has_take = cube_subdecisions(entry)
    return cube_errors(entry["equity_nd"], entry["equity_dt"], entry["equity_dp"],
                       has_double, has_take, pick), False


def errors_vs_xg(xg: dict, pick) -> tuple[list[float], bool]:
    """Per-sub-decision errors against XG's evaluation."""
    if xg["kind"] == "checker":
        vals = {b: e for b, e, _ in xg["moves"]}
        best = max(vals.values())
        eq = vals.get(tuple(pick))
        if eq is None:
            return [best - min(vals.values())], True
        return [best - eq], False
    nd, dt, dp = xg["nd"], xg["dt"], xg["dp"]
    pseudo = {"cube_owner": "centered", "equity_nd": nd, "equity_dt": dt, "equity_dp": dp,
              "should_double_ref": min(dt, dp) > nd, "is_beaver": dt < 0}
    has_double, has_take = cube_subdecisions(pseudo)
    return cube_errors(nd, dt, dp, has_double, has_take, pick), False


def xg_pick(xg: dict):
    if xg["kind"] == "checker":
        return list(xg["moves"][0][0])
    return {"should_double": xg["flag_double"] > 0, "should_take": xg["dt"] <= xg["dp"]}


def summarize(per_dec: list[list[float]], outside: int) -> dict:
    flat = [e for d in per_dec for e in d]
    n = len(flat)
    return {"pr": sum(flat) / n * PR_MULTIPLIER if n else float("nan"), "n": n,
            "blunders": sum(e > BLUNDER_THRESHOLD for e in flat), "outside": outside}


def paired(a: list[list[float]], b: list[list[float]]) -> tuple[float, float]:
    """Paired PR difference b - a per sub-decision, with its standard error."""
    diffs = [y - x for da, db in zip(a, b) for x, y in zip(da, db)]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean * PR_MULTIPLIER, math.sqrt(var / n) * PR_MULTIPLIER


def cmd_score(args) -> None:
    from bgsage.weights import PRODUCTION_MODEL

    model = args.model or PRODUCTION_MODEL
    summary = {"xg_dir": str(args.xg_dir), "model": model, "folders": {}}
    for folder in args.folders:
        xg = harvest_xg(args.xg_dir, args.gen_dir, folder)
        ref = {e["key"]: e for e in exported_entries(folder)}
        players: dict[str, dict] = {"XG #1": {k: xg_pick(v) for k, v in xg.items()}}
        for level in args.levels:
            p = picks_path(args.scores_dir, model, level, folder)
            if not p.exists():
                print(f"  (no Sage picks for {level}: run 'picks' first)")
                continue
            players[f"{model} {level}"] = {
                r["key"]: r["pick"] for r in map(json.loads, p.read_text().splitlines())
                if r.get("pick") is not None}
        # Only decisions every player and XG cover, so all rows are paired.
        keys = [k for k in ref if k in xg and all(k in pk for pk in players.values())]
        kinds = Counter(ref[k]["kind"] for k in keys)
        levels = Counter(xg[k]["level"] for k in keys)
        print(f"\n{folder}: {len(keys)} of {len(ref)} decisions scored "
              f"({kinds['checker']} checker, {kinds['cube']} cube); XG levels {dict(levels)}")
        errs: dict[str, dict[str, dict[str, list]]] = {}
        table = {}
        for name, pk in players.items():
            errs[name] = {}
            row = {}
            for refname, fn in (("rollout", lambda k, p: errors_vs_rollout(ref[k], p)),
                                ("XG", lambda k, p: errors_vs_xg(xg[k], p))):
                for part in ("all", "checker", "cube"):
                    sel = [k for k in keys if part == "all" or ref[k]["kind"] == part]
                    res = [fn(k, pk[k]) for k in sel]
                    errs[name][f"{refname}/{part}"] = [r[0] for r in res]
                    row[f"{refname}/{part}"] = summarize([r[0] for r in res],
                                                         sum(r[1] for r in res))
            table[name] = row
        print(f"  {'player':16}{'PR vs rollout':>14}{'checker':>9}{'cube':>7}{'blund':>7}"
              f"{'PR vs XG':>11}{'checker':>9}{'cube':>7}{'blund':>7}{'outside XG':>12}")
        for name, row in table.items():
            r, x = row["rollout/all"], row["XG/all"]
            print(f"  {name:16}{r['pr']:>14.2f}{row['rollout/checker']['pr']:>9.2f}"
                  f"{row['rollout/cube']['pr']:>7.2f}{r['blunders']:>7}"
                  f"{x['pr']:>11.2f}{row['XG/checker']['pr']:>9.2f}{row['XG/cube']['pr']:>7.2f}"
                  f"{x['blunders']:>7}{row['XG/checker']['outside']:>12}")
        sage = [n for n in players if n != "XG #1"]
        diffs = {}
        if len(sage) > 1:
            print("  paired level differences (PR, later minus earlier, +- 1 SE):")
            for i in range(len(sage)):
                for j in range(i + 1, len(sage)):
                    a, b = sage[i], sage[j]
                    cells = []
                    for refname in ("rollout", "XG"):
                        for part in ("all", "checker", "cube"):
                            m, se = paired(errs[a][f"{refname}/{part}"],
                                           errs[b][f"{refname}/{part}"])
                            diffs[f"{b} - {a} | {refname}/{part}"] = (m, se)
                            if part == "all":
                                cells.append(f"vs {refname} {m:+.2f} +- {se:.2f}")
                    print(f"    {b.split()[-1]} - {a.split()[-1]}: " + ";  ".join(cells))
        summary["folders"][folder] = {"n": len(keys), "xg_levels": dict(levels),
                                      "table": table,
                                      "paired": {k: {"pr_diff": m, "se": s}
                                                 for k, (m, s) in diffs.items()}}
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    out = args.scores_dir / f"sage_vs_xg_{args.label}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("picks", cmd_picks), ("score", cmd_score)):
        p = sub.add_parser(name)
        p.add_argument("--folders", nargs="+", default=["snake", "snake s11play"])
        p.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
        p.add_argument("--model", default=None, help="default: the production model")
        p.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
        p.set_defaults(func=fn)
        if name == "picks":
            p.add_argument("--threads", type=int, default=0)
        else:
            p.add_argument("--xg-dir", type=Path, required=True,
                           help="the XG-analysed copy of the generated tree")
            p.add_argument("--gen-dir", type=Path, default=fx.DEFAULT_OUT_DIR)
            p.add_argument("--label", default="xg",
                           help="suffix of the summary file, e.g. the XG level")
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
