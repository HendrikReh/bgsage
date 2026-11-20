# SPDX-License-Identifier: MPL-2.0
"""Per-decision pure-race errors on the Benchmark PR rollout sample.

Scores every pure-race decision of the sample (``--decision-step``, default 20)
for the production model at 1-ply and 2-ply, with and without the bearoff
database, and for GNUbg at 1-ply and 2-ply, then prints PR per subset:

* bearoff — the pre-move position is in the one-sided bearoff database
  (both sides home);
* race outside the database, split by whether any candidate lands in it.

The per-decision errors (equity units) go to a JSON file for further digging;
``scripts/race_net_swap.py`` reads it. Findings from 2026-11-18 are in
CLAUDE.md, "Benchmark PR rollout sample (this fork)" -> "The pure-race gap".

Usage:
    python scripts/race_gap_analysis.py
    python scripts/race_gap_analysis.py --model stage5 --out logs/race_gap/stage5.json
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "build", _PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
import score_benchmark_pr as sp  # noqa: E402
import score_benchmark_pr_gnubg as sg  # noqa: E402
from neongammon.weights import PRODUCTION_MODEL, WeightConfig, bearoff_db_path  # noqa: E402

DEFAULT_OUT = _PROJECT_ROOT / "logs" / "race_gap" / "race_gap.json"
KEYS = ("board", "dice", "candidates", "rollout_equities")
CONFIGS = ("sage1", "sage1_db", "sage2", "sage2_db", "gnubg1", "gnubg2")


def _gnubg_err(args):
    dec, plies = args
    r = sg.score_decision_gnubg(dec, plies)
    return None if r is None else (r[1], r[2])


def collect(model, decision_step, workers):
    bgbot_cpp.init_escape_tables()
    decs, _ = sp.load_benchmark_data(decision_step)
    race = [d for d in decs if len(d["candidates"]) >= 2
            and bgbot_cpp.classify_game_plan(d["board"]) == "purerace"]
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path() or ""):
        sys.exit("ERROR: bearoff database not found (scripts/generate_bearoff_db.py)")
    for d in race:
        d["bearoff"] = bool(db.is_bearoff(d["board"]))
        d["cand_bearoff"] = sum(bool(db.is_bearoff(c)) for c in d["candidates"])

    w = WeightConfig.from_model(model)
    args = (w.strategy_type, w.weight_paths_list, w.hidden_sizes_list)
    base = bgbot_cpp.create_strategy(*args)
    base_db = bgbot_cpp.BearoffStrategy(base, db)
    m2 = bgbot_cpp.create_multipy(*args, n_plies=2)
    m2_db = bgbot_cpp.create_multipy(*args, n_plies=2)
    bgbot_cpp.multipy_set_bearoff_db(m2_db, db)
    strategies = {"sage1": (base, None), "sage1_db": (base_db, None),
                  "sage2": (m2, base), "sage2_db": (m2_db, base_db)}
    for name, (strat, filt) in strategies.items():
        for d in race:
            r = bgbot_cpp.score_benchmark_pr_strategy([{k: d[k] for k in KEYS}], strat, filt,
                                                      n_threads=1)
            d[name] = r["total_error"] if r["total_decisions"] else None
        print(f"  {name} done", flush=True)

    plain = [{k: d[k] for k in KEYS} for d in race]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for plies in (1, 2):
            res = list(pool.map(_gnubg_err, [(d, plies) for d in plain], chunksize=8))
            for d, r in zip(race, res):
                d[f"gnubg{plies}"] = None if r is None else r[0]
                d[f"gnubg{plies}_outside"] = None if r is None else r[1]
            print(f"  gnubg{plies} done", flush=True)
    return race


def summarize(race):
    subsets = {
        "all race": race,
        "bearoff (pre-move)": [d for d in race if d["bearoff"]],
        "race, not bearoff": [d for d in race if not d["bearoff"]],
        "  ..some cands bearoff": [d for d in race if not d["bearoff"] and d["cand_bearoff"]],
        "  ..no cand bearoff": [d for d in race if not d["bearoff"] and not d["cand_bearoff"]],
    }
    print(f"{'subset':26}{'n':>5}" + "".join(f"{k:>10}" for k in CONFIGS))
    for name, ds in subsets.items():
        row = f"{name:26}{len(ds):>5}"
        for k in CONFIGS:
            v = [d[k] for d in ds if d.get(k) is not None]
            row += f"{(sum(v) / len(v) * 500 if v else float('nan')):>10.2f}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=PRODUCTION_MODEL)
    ap.add_argument("--decision-step", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12, help="parallel GNUbg processes")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary-only", action="store_true",
                    help="print the table from an existing --out file")
    args = ap.parse_args()

    if not args.summary_only:
        race = collect(args.model, args.decision_step, args.workers)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(race))
        print(f"  wrote {args.out}")
    summarize(json.loads(args.out.read_text()))


if __name__ == "__main__":
    main()
