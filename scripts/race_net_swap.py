# SPDX-License-Identifier: MPL-2.0
"""Score the production model with different pure-race nets swapped in.

Reads the race decisions written by ``scripts/race_gap_analysis.py`` and
scores them at 1-ply / 2-ply (bearoff database on) with each pure-race net in
NETS in place of the model's own. Caveat: the Benchmark PR reference is Stage
5's rollout, played with Stage 5's pure-race net, so that net is favoured.

Usage:
    python scripts/race_net_swap.py
    python scripts/race_net_swap.py --net "my net=sl_new_purerace.weights.best:200"
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
from neongammon.weights import PRODUCTION_MODEL, WeightConfig, bearoff_db_path  # noqa: E402

DEFAULT_IN = _PROJECT_ROOT / "logs" / "race_gap" / "race_gap.json"
KEYS = ("board", "dice", "candidates", "rollout_equities")
NETS = {
    "s6 100h (stage6-11)": ("sl_s9_purerace.weights.best", 100),
    "stage3 120h": ("sl_purerace.weights.best", 120),
    "s4 120h": ("sl_s4_purerace.weights.best", 120),
    "s5 200h": ("sl_s5_purerace.weights.best", 200),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=PRODUCTION_MODEL)
    ap.add_argument("--decisions", type=Path, default=DEFAULT_IN)
    ap.add_argument("--net", action="append", default=[],
                    help='extra net "label=file_in_models:hidden" (repeatable)')
    args = ap.parse_args()

    nets = dict(NETS)
    for spec in args.net:
        label, rest = spec.split("=", 1)
        fname, nh = rest.rsplit(":", 1)
        nets[label] = (fname, int(nh))

    bgbot_cpp.init_escape_tables()
    race = json.loads(args.decisions.read_text())
    subsets = {
        "race, not bearoff": [{k: d[k] for k in KEYS} for d in race if not d["bearoff"]],
        "..no cand bearoff": [{k: d[k] for k in KEYS} for d in race
                              if not d["bearoff"] and d["cand_bearoff"] == 0],
        "all race": [{k: d[k] for k in KEYS} for d in race],
    }
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path() or ""):
        sys.exit("ERROR: bearoff database not found (scripts/generate_bearoff_db.py)")
    w = WeightConfig.from_model(args.model)

    print(f"{'pure-race net':22}" + "".join(f"{s:>22}" for s in subsets) + "   (1-ply / 2-ply)")
    for label, (fname, nh) in nets.items():
        paths, hidden = list(w.weight_paths_list), list(w.hidden_sizes_list)
        paths[0], hidden[0] = str(_PROJECT_ROOT / "models" / fname), nh
        base = bgbot_cpp.BearoffStrategy(
            bgbot_cpp.create_strategy(w.strategy_type, paths, hidden), db)
        m2 = bgbot_cpp.create_multipy(w.strategy_type, paths, hidden, n_plies=2)
        bgbot_cpp.multipy_set_bearoff_db(m2, db)
        row = f"{label:22}"
        for ds in subsets.values():
            r1 = bgbot_cpp.score_benchmark_pr_strategy(ds, base)["overall_pr"]
            r2 = bgbot_cpp.score_benchmark_pr_strategy(ds, m2, base)["overall_pr"]
            row += f"{r1:>12.2f} / {r2:<7.2f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
