# SPDX-License-Identifier: MPL-2.0
"""Per-call latency of BgBotAnalyzer's cheap paths (cube_action, checker_play).

Prints one JSON line: median ms per call on fixed random positions, one
persistent analyzer per level. Used to measure PR #3 (reused cube strategy,
serial 1-ply conversion); the results are in CLAUDE.md, "Analyzer call
overhead (PR #3)".

To compare two engine versions, build the other one in a worktree and point
--root at it (each run loads that checkout's build/ and python/), alternating
runs so both see the same machine load:

    git worktree add --detach build/bench_old <commit>   # then build it there
    for i in 1 2 3; do
      python scripts/bench_analyzer_overhead.py --root build/bench_old --label old
      python scripts/bench_analyzer_overhead.py --label new
    done
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--root", type=Path, default=_PROJECT_ROOT,
                help="checkout whose build/ and python/ to load (default: this repo)")
ap.add_argument("--label", default="this")
args = ap.parse_args()
root = args.root.resolve()
sys.path[:0] = [str(root / "build"), str(root / "python")]

from neongammon import (BgBotAnalyzer, STARTING_BOARD, check_game_over,  # noqa: E402
                    flip_board, possible_moves)

rng = random.Random(11)
positions = []
while len(positions) < 60:
    b = list(STARTING_BOARD)
    for _ in range(rng.randint(4, 60)):
        moves = possible_moves(b, rng.randint(1, 6), rng.randint(1, 6))
        if moves:
            b = list(rng.choice(moves))
        if check_game_over(b):
            break
        b = flip_board(b)
    if not check_game_over(b):
        positions.append((b, rng.randint(1, 6), rng.randint(1, 6)))


def timed(fn, items):
    fn(*items[0])  # warm-up
    ts = []
    for it in items:
        t0 = time.perf_counter()
        fn(*it)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


out = {"label": args.label, "root": str(root)}
for level, n in (("1ply", 60), ("2ply", 60), ("3ply", 20)):
    a = BgBotAnalyzer(eval_level=level)
    out[f"cube {level}"] = timed(
        lambda b, d1, d2: a.cube_action(b, cube_value=1, cube_owner="centered"), positions[:n])
    if level != "3ply":
        out[f"checker {level}"] = timed(lambda b, d1, d2: a.checker_play(b, d1, d2), positions[:n])
a = BgBotAnalyzer(eval_level="1ply", parallel_threads=1)
out["checker 1ply, 1 thread"] = timed(lambda b, d1, d2: a.checker_play(b, d1, d2), positions)
print(json.dumps(out))
