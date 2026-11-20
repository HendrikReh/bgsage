#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Score Neongammon's picks on a folder benchmark against XG AND against the
folder's rollout reference, side by side, on the same decisions.

Why: on the snake folder the two rollout references disagree about Neongammon's
levels -- stage11 3-ply is the worst level under the S9-play reference and the
best under the S11-play twin (CLAUDE.md, "Stage 11s"). XG is an independent
referee for both decision sets.

Workflow
--------
1. ``export_folder_benchmark_xg.py generate --folders snake "snake s11play"``
   writes the one-decision ``.xg`` shards (``data/backgame_xg/<slug>/``).
2. ``score_folder_vs_xg.py picks --folders snake "snake s11play"`` plays every
   exported decision with Neongammon at 1/2/3-ply and caches the picks in
   ``data/backgame_xg_scores/sage_<model>_<level>_<slug>_picks.jsonl``
   (resumable: decisions already in the file are skipped). Independent of XG,
   so it can run while XG works.
3. A copy of ``data/backgame_xg/`` is Batch-Analyzed in XG ("Save games after
   analyze" ON) at the chosen level.
4. ``score_folder_vs_xg.py score --xg-dir <copy>`` grades every cached Neongammon
   level and XG's own #1 against both references and prints PRs plus the
   paired level differences with their standard errors.

How XG is used as a reference
-----------------------------
* Checker: XG's CUBEFUL equity (``eval[6]``; the folder decisions are money
  games with Jacoby + Beaver, as is the template header, and the rollout
  references are cubeful too) -- but ONLY for moves XG took to the decision's
  full level (its #1's level). XG leaves the other candidates at the depth
  its filter dropped them, and in snake positions those values are not on
  the full level's scale: a 3-ply value regularly beats the 4-ply #1, so
  grading against the stored maximum charged XG's own #1 PR 61. XG always
  analyses the PLAYED move at the full level (every played move of the first
  run sat there), so step 5 below re-asks XG with Neongammon's pick played.
  A pick that is in neither set is not graded and is counted ("ungraded").

5. ``score_folder_vs_xg.py complete --xg-dir <copy>`` writes one-decision
   games with each Neongammon pick that XG left below its full level as the played
   move (``data/backgame_xg_completion/``); XG Batch-Analyzes them like step
   3, and ``score --completion-dir <analysed copy>`` folds them in.
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
from neongammon import xg_file as xf  # noqa: E402
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
    from neongammon import BgBotAnalyzer
    from neongammon.weights import PRODUCTION_MODEL, WeightConfigPair

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
                top = rec["moves"][0]["level"]
                out[sc["key"]] = {
                    "kind": "checker", "level": xf.player_level_label(top),
                    "best": float(rec["moves"][0]["eval"][6]),
                    "deep": {tuple(m["board"]): float(m["eval"][6])
                             for m in rec["moves"] if m["level"] == top},
                    "best_board": tuple(rec["moves"][0]["board"])}
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


def harvest_completion(comp_dir: Path | None, folder: str) -> dict[tuple, float]:
    """(key, played board) -> XG's full-level error of that played move, from
    the completion shards (``<slug>__compl_shard_NNN.xg`` + sidecars)."""
    out: dict[tuple, float] = {}
    if comp_dir is None:
        return out
    slug = fx.folder_slug(folder)
    for path in sorted(comp_dir.glob(f"{slug}__compl_shard_*.xg")):
        sidecar = fx.load_sidecar(path, None)
        tx = xf.XgArchive.load(path).get("temp.xg")
        games = fx.iter_games(tx)
        if len(games) != len(sidecar):
            raise ValueError(f"{path}: {len(games)} games, {len(sidecar)} sidecar rows")
        for sc, g in zip(sidecar, games):
            rec = xf.parse_move_record(tx, g["move"][0])
            if list(rec["mover_board"]) != list(sc["board"]):
                raise ValueError(f"{path} game {sc['game_index']}: board mismatch")
            if rec["n_moves"] <= 0 or not rec["moves"]:
                continue
            top = rec["moves"][0]["level"]
            played = tuple(sc["played_board"])
            eq = next((float(m["eval"][6]) for m in rec["moves"]
                       if tuple(m["board"]) == played and m["level"] == top), None)
            if eq is None:
                raise ValueError(f"{path} game {sc['game_index']}: XG did not take the "
                                 f"played move to its full level")
            out[(sc["key"], played)] = max(0.0, float(rec["moves"][0]["eval"][6]) - eq)
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


def errors_vs_xg(xg: dict, pick, key: str = "", comp: dict | None = None
                 ) -> tuple[list[float], bool]:
    """Per-sub-decision errors against XG's evaluation; ``([], True)`` for a
    checker pick XG never took to its full level."""
    if xg["kind"] == "checker":
        eq = xg["deep"].get(tuple(pick))
        if eq is not None:
            return [max(0.0, xg["best"] - eq)], False
        err = (comp or {}).get((key, tuple(pick)))
        if err is not None:
            return [err], False
        return [], True
    nd, dt, dp = xg["nd"], xg["dt"], xg["dp"]
    pseudo = {"cube_owner": "centered", "equity_nd": nd, "equity_dt": dt, "equity_dp": dp,
              "should_double_ref": min(dt, dp) > nd, "is_beaver": dt < 0}
    has_double, has_take = cube_subdecisions(pseudo)
    return cube_errors(nd, dt, dp, has_double, has_take, pick), False


def xg_pick(xg: dict):
    if xg["kind"] == "checker":
        return list(xg["best_board"])
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
    from neongammon.weights import PRODUCTION_MODEL

    model = args.model or PRODUCTION_MODEL
    summary = {"xg_dir": str(args.xg_dir), "model": model, "folders": {}}
    for folder in args.folders:
        xg = harvest_xg(args.xg_dir, args.gen_dir, folder)
        comp = harvest_completion(args.completion_dir, folder)
        ref = {e["key"]: e for e in exported_entries(folder)}
        players: dict[str, dict] = {"XG #1": {k: xg_pick(v) for k, v in xg.items()}}
        for level in args.levels:
            p = picks_path(args.scores_dir, model, level, folder)
            if not p.exists():
                print(f"  (no Neongammon picks for {level}: run 'picks' first)")
                continue
            players[f"{model} {level}"] = {
                r["key"]: r["pick"] for r in map(json.loads, p.read_text().splitlines())
                if r.get("pick") is not None}
        # Only decisions every player covers and XG grades for every player,
        # so all rows are paired.
        cover = [k for k in ref if k in xg and all(k in pk for pk in players.values())]
        ungraded = {n: sum(errors_vs_xg(xg[k], pk[k], k, comp)[1] for k in cover)
                    for n, pk in players.items()}
        keys = [k for k in cover
                if not any(errors_vs_xg(xg[k], pk[k], k, comp)[1] for pk in players.values())]
        if any(ungraded.values()):
            print(f"  {folder}: decisions left out because XG never took a player's pick to "
                  f"its full level: {ungraded} (run 'complete', then XG, then --completion-dir)")
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
                                ("XG", lambda k, p: errors_vs_xg(xg[k], p, k, comp))):
                for part in ("all", "checker", "cube"):
                    sel = [k for k in keys if part == "all" or ref[k]["kind"] == part]
                    res = [fn(k, pk[k]) for k in sel]
                    errs[name][f"{refname}/{part}"] = [r[0] for r in res]
                    row[f"{refname}/{part}"] = summarize([r[0] for r in res],
                                                         sum(r[1] for r in res))
            table[name] = row
        print(f"  {'player':16}{'PR vs rollout':>14}{'checker':>9}{'cube':>7}{'blund':>7}"
              f"{'PR vs XG':>11}{'checker':>9}{'cube':>7}{'blund':>7}")
        for name, row in table.items():
            r, x = row["rollout/all"], row["XG/all"]
            print(f"  {name:16}{r['pr']:>14.2f}{row['rollout/checker']['pr']:>9.2f}"
                  f"{row['rollout/cube']['pr']:>7.2f}{r['blunders']:>7}"
                  f"{x['pr']:>11.2f}{row['XG/checker']['pr']:>9.2f}{row['XG/cube']['pr']:>7.2f}"
                  f"{x['blunders']:>7}")
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
        summary["folders"][folder] = {"n": len(keys), "n_covered": len(cover),
                                      "ungraded": ungraded, "xg_levels": dict(levels),
                                      "table": table,
                                      "paired": {k: {"pr_diff": m, "se": s}
                                                 for k, (m, s) in diffs.items()}}
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    out = args.scores_dir / f"sage_vs_xg_{args.label}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


COMPLETION_SHARD = "{slug}__compl_shard_{n:03d}.xg"
DEFAULT_COMPLETION_DIR = _PROJECT_ROOT / "data" / "backgame_xg_completion"


def cmd_complete(args) -> None:
    """One-decision games with every Neongammon checker pick XG left below its full
    level as the played move, so XG analyses exactly those at full depth."""
    from neongammon.weights import PRODUCTION_MODEL
    from export_pasko_benchmark_xg import _templates, derive_half_moves, write_xg

    model = args.model or PRODUCTION_MODEL
    tpl = _templates(args.template)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*__compl_shard_*"):
        stale.unlink()
    total = 0
    for folder in args.folders:
        xg = harvest_xg(args.xg_dir, args.gen_dir, folder)
        ref = {e["key"]: e for e in exported_entries(folder)}
        need: dict[tuple, dict] = {}
        for level in args.levels:
            p = picks_path(args.scores_dir, model, level, folder)
            for r in map(json.loads, p.read_text().splitlines()):
                k = r["key"]
                if r["kind"] != "checker" or r["pick"] is None or k not in xg:
                    continue
                pick = tuple(r["pick"])
                if pick in xg[k]["deep"] or (k, pick) in need:
                    continue
                e = ref[k]
                d1, d2 = e["dice"]
                board = list(e["board"])
                need[(k, pick)] = {
                    "key": k, "folder": folder, "kind": "checker", "board": board,
                    "cube_value": e["cube_value"], "cube_owner": e["cube_owner"],
                    "cube_code": fx.cube_code(e["cube_value"], e["cube_owner"]),
                    "double": fx.NO_CUBE_DECISION, "dice": [d1, d2], "post": list(pick),
                    "half_moves": derive_half_moves(board, list(pick), d1, d2)}
        games = list(need.values())
        slug = fx.folder_slug(folder)
        n_shards = (len(games) + args.per_file - 1) // args.per_file
        for n in range(n_shards):
            chunk = games[n * args.per_file:(n + 1) * args.per_file]
            path = args.out_dir / COMPLETION_SHARD.format(slug=slug, n=n)
            write_xg(args.template, fx.build_shard_stream(tpl, chunk), path)
            with open(str(path) + fx.SIDECAR_SUFFIX, "w", encoding="utf-8") as f:
                for gi, g in enumerate(chunk):
                    f.write(json.dumps(fx.sidecar_row(gi, g), separators=(",", ":")) + "\n")
            fx.verify_shard(path, chunk)
        total += len(games)
        print(f"{folder}: {len(games)} Neongammon picks below XG's full level -> {n_shards} "
              f"shard(s), verified")
    print(f"\n{total} games in {args.out_dir}. Batch-Analyze that folder in XG at the SAME "
          f"level as the first run ('Save games after analyze' ON), then:\n"
          f"  score_folder_vs_xg.py score --xg-dir <first run> --completion-dir {args.out_dir}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("picks", cmd_picks), ("score", cmd_score), ("complete", cmd_complete)):
        p = sub.add_parser(name)
        p.add_argument("--folders", nargs="+", default=["snake", "snake s11play"])
        p.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
        p.add_argument("--model", default=None, help="default: the production model")
        p.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
        p.set_defaults(func=fn)
        if name == "picks":
            p.add_argument("--threads", type=int, default=0)
        elif name == "complete":
            p.add_argument("--xg-dir", type=Path, required=True,
                           help="the XG-analysed copy of the generated tree")
            p.add_argument("--gen-dir", type=Path, default=fx.DEFAULT_OUT_DIR)
            p.add_argument("--out-dir", type=Path, default=DEFAULT_COMPLETION_DIR)
            p.add_argument("--template", type=Path, default=fx.DEFAULT_TEMPLATE)
            p.add_argument("--per-file", type=int, default=fx.DEFAULT_PER_FILE)
        else:
            p.add_argument("--completion-dir", type=Path, default=None,
                           help="XG-analysed completion shards from 'complete'")
            p.add_argument("--xg-dir", type=Path, required=True,
                           help="the XG-analysed copy of the generated tree")
            p.add_argument("--gen-dir", type=Path, default=fx.DEFAULT_OUT_DIR)
            p.add_argument("--label", default="xg",
                           help="suffix of the summary file, e.g. the XG level")
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
