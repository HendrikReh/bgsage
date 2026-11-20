"""4P optimization benchmark.

Reference position: post-move state of 8/6 8/3 on the gammon-loss-by-roll trial.
Calls `post_move_analytics` at 4P (money + match) and `cube_action` at 4P
(money + match) using 16 threads. Reports wall time and the full numerical
output so accuracy regressions are easy to spot.

Outputs JSON to `data/bench_4p_<label>.json` so multiple labels can be
compared after each optimization attempt.

Usage:
    python scripts/bench_4p_opt.py --label baseline
    python scripts/bench_4p_opt.py --label idea1 --iterations 5
    python scripts/bench_4p_opt.py --compare baseline idea1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "build"))
sys.path.insert(0, str(REPO / "python"))

import bgbot_cpp  # noqa: E402
from bgsage import BgBotAnalyzer  # noqa: E402


# Post-move position from the 8/6 8/3 play of dice 5-2 on the reference
# pre-move board [0,2,2,1,2,0,2,0,2,1,0,0,-4,2,0,0,0,-3,0,-2,-4,1,-2,0,0,0]
POST = [0, 2, 2, 1, 2, 0, 2, 0, 0, 1, 0, 0,
        -4, 2, 0, 0, 0, -3, 0, -2, -4, 1, -2, 0, 0, 0]
# Apply 8->6 and 8->3
POST = [0, 2, 2, 1, 2, 0, 2, 0, 2, 1, 0, 0,
        -4, 2, 0, 0, 0, -3, 0, -2, -4, 1, -2, 0, 0, 0]
POST[8] = 0   # remove 2 checkers from point 8
POST[6] = 3   # added 1 from 8/6
POST[3] = 2   # added 1 from 8/3 (was 1)

# Pre-move board for cube_action
PRE_MOVE = [0, 2, 2, 1, 2, 0, 2, 0, 2, 1, 0, 0,
            -4, 2, 0, 0, 0, -3, 0, -2, -4, 1, -2, 0, 0, 0]


@dataclass
class CallResult:
    label: str
    wall: float
    counters: dict
    payload: dict  # full numerical output for accuracy verification


def reset_and_call(fn, *args, **kwargs):
    bgbot_cpp.reset_cubeful_counters()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    counters = dict(bgbot_cpp.get_cubeful_counters())
    return out, wall, counters


def run_post_move_money(analyzer):
    r = analyzer.post_move_analytics(
        POST, cube_owner="opponent", cube_value=2, jacoby=True,
    )
    return {
        "win": r.probs.win,
        "gw": r.probs.gammon_win,
        "bw": r.probs.backgammon_win,
        "gl": r.probs.gammon_loss,
        "bl": r.probs.backgammon_loss,
        "cubeless_equity": r.cubeless_equity,
        "cubeful_equity": r.cubeful_equity,
    }


def run_post_move_match(analyzer):
    r = analyzer.post_move_analytics(
        POST, cube_owner="opponent", cube_value=2,
        away1=4, away2=1, is_crawford=False,
    )
    return {
        "win": r.probs.win,
        "gw": r.probs.gammon_win,
        "bw": r.probs.backgammon_win,
        "gl": r.probs.gammon_loss,
        "bl": r.probs.backgammon_loss,
        "cubeless_equity": r.cubeless_equity,
        "cubeful_equity": r.cubeful_equity,
    }


def run_cube_action_money(analyzer):
    r = analyzer.cube_action(
        PRE_MOVE, cube_value=1, cube_owner="centered",
    )
    return {
        "win": r.probs.win, "gw": r.probs.gammon_win, "bw": r.probs.backgammon_win,
        "gl": r.probs.gammon_loss, "bl": r.probs.backgammon_loss,
        "cubeless_equity": r.cubeless_equity,
        "equity_nd": r.equity_nd,
        "equity_dt": r.equity_dt,
        "equity_dp": r.equity_dp,
    }


def run_cube_action_match(analyzer):
    r = analyzer.cube_action(
        PRE_MOVE, cube_value=1, cube_owner="centered",
        away1=4, away2=1, is_crawford=False,
    )
    return {
        "win": r.probs.win, "gw": r.probs.gammon_win, "bw": r.probs.backgammon_win,
        "gl": r.probs.gammon_loss, "bl": r.probs.backgammon_loss,
        "cubeless_equity": r.cubeless_equity,
        "equity_nd": r.equity_nd,
        "equity_dt": r.equity_dt,
        "equity_dp": r.equity_dp,
    }


def bench(label: str, iterations: int, threads: int):
    analyzer = BgBotAnalyzer(eval_level="4ply", cubeful=True,
                             parallel_threads=threads)

    print(f"[{label}] warmup ...", flush=True)
    # Warmup once to populate caches; result discarded.
    run_post_move_match(analyzer)

    cases = [
        ("post_move_money", run_post_move_money),
        ("post_move_match", run_post_move_match),
        ("cube_action_money", run_cube_action_money),
        ("cube_action_match", run_cube_action_match),
    ]

    out: dict[str, Any] = {"label": label, "threads": threads, "cases": {}}

    for case_name, runner in cases:
        walls = []
        last_payload = None
        last_counters = None
        for i in range(iterations):
            payload, wall, counters = reset_and_call(runner, analyzer)
            walls.append(wall)
            last_payload = payload
            last_counters = counters
            print(f"[{label}] {case_name} iter {i+1}/{iterations}: "
                  f"{wall:.3f}s leaf={counters['leaf_count']} "
                  f"internal={counters['internal_count']} "
                  f"cache_hit={counters['cache_hit_count']} "
                  f"move_gen={counters['move_gen_count']}",
                  flush=True)
        out["cases"][case_name] = {
            "walls": walls,
            "best": min(walls),
            "mean": sum(walls) / len(walls),
            "counters": last_counters,
            "payload": last_payload,
        }

    return out


def compare(labels: list[str], data_dir: Path):
    """Print a side-by-side comparison of the given labels."""
    results = {}
    for label in labels:
        p = data_dir / f"bench_4p_{label}.json"
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            sys.exit(1)
        results[label] = json.loads(p.read_text())

    base_label = labels[0]
    base = results[base_label]

    # Show speedup vs baseline per case
    print(f"\n=== Speed comparison (vs {base_label}) ===")
    cases = list(base["cases"].keys())
    cols = ["case"] + labels
    widths = [22] + [16] * len(labels)
    header = " ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * sum(widths))
    for case_name in cases:
        row = [case_name]
        base_wall = base["cases"][case_name]["best"]
        for label in labels:
            wall = results[label]["cases"][case_name]["best"]
            speedup = base_wall / wall if wall > 0 else float("inf")
            row.append(f"{wall:.3f}s ({speedup:.2f}x)")
        print(" ".join(f"{c:<{w}}" for c, w in zip(row, widths)))

    # Show counter comparison per case (helps spot what changed)
    print(f"\n=== Counter comparison ===")
    keys = ["leaf_count", "internal_count", "cache_hit_count", "move_gen_count"]
    for case_name in cases:
        print(f"\n  {case_name}")
        cols = ["counter"] + labels
        widths = [18] + [16] * len(labels)
        print(" ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
        print("-" * sum(widths))
        for k in keys:
            row = [k]
            for label in labels:
                v = results[label]["cases"][case_name]["counters"][k]
                row.append(f"{v:>14}")
            print(" ".join(f"{c:<{w}}" for c, w in zip(row, widths)))

    # Accuracy: max abs delta in payloads
    print(f"\n=== Accuracy deltas (max |delta| vs {base_label}) ===")
    for case_name in cases:
        base_payload = base["cases"][case_name]["payload"]
        print(f"\n  {case_name}")
        # Show baseline values
        kw_cols = ["key", base_label]
        widths = [22, 14] + [14 for _ in labels[1:]]
        for label in labels[1:]:
            kw_cols.append(label)
        print(" ".join(f"{c:<{w}}" for c, w in zip(kw_cols, widths)))
        print("-" * sum(widths))
        for k, base_v in base_payload.items():
            row = [k, f"{base_v:+.6f}"]
            for label in labels[1:]:
                v = results[label]["cases"][case_name]["payload"][k]
                d = v - base_v
                row.append(f"{d:+.6f}")
            print(" ".join(f"{c:<{w}}" for c, w in zip(row, widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, default=None,
                    help="Save results to data/bench_4p_<label>.json")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--compare", nargs="+", default=None,
                    help="Compare two or more labels and exit")
    args = ap.parse_args()

    data_dir = REPO / "data"
    data_dir.mkdir(exist_ok=True)

    if args.compare:
        compare(args.compare, data_dir)
        return

    if args.label is None:
        ap.error("--label is required when not using --compare")

    print(f"Running bench_4p_opt: label={args.label} threads={args.threads}")
    print(f"Position: {POST}\n")
    result = bench(args.label, args.iterations, args.threads)
    out_path = data_dir / f"bench_4p_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
