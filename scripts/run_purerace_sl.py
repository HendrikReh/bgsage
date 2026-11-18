# SPDX-License-Identifier: MPL-2.0
"""SL training for the pure-race NN on our own rollout targets.

Data: ``data/purerace-sage-train-rollout`` (positions from
``scripts/harvest_race_positions.py``, rolled out by
``scripts/modal_rollout.py``: Stage 9 play, 1,296 trials, 3-ply, VR, bearoff
DB). The holdout is ``data/purerace-sage-benchmark-rollout``, whose positions
come from separate games (every 10th). Neither holds a bearoff-DB position:
the engine plays those from the database.

ER = mean |1-ply equity - rollout equity| x 1000 on the holdout; the best-ER
weights are kept. The net warm-starts from an existing pure-race net (SL
from random init lands in poor minima; see CLAUDE.md "Training Pipeline").
Before training, the holdout ER of the start net is printed, and of the
rollout targets' own noise (their mean standard error x 1000 x sqrt(2/pi),
the ER a perfect net would still show).

The decisive check is not the holdout ER but the decisions: score the
result with ``export_benchmark_pr_xg.py sage --race-net`` against XG's
analysis of the Benchmark PR races.

Usage:
    python scripts/run_purerace_sl.py --init-from models/sl_s9_purerace.weights.best --hidden 100 --tag h100
    python scripts/run_purerace_sl.py --init-from models/sl_s5_purerace.weights.best --hidden 200 --tag h200
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
from bgsage import sl_train  # noqa: E402

N_INPUTS = 196
BATCH = 4096
#: (epochs, alpha): one epoch is one pass over the ~330k training rows.
PHASES = [(1000, 6.3), (2000, 2.0), (2000, 0.63)]
CHUNK = 100
_DATA = _PROJECT_ROOT / "data"
_OUT = _PROJECT_ROOT / "models" / "race_diag"


def load_rollout(path: Path):
    boards, probs = [], []
    for line in path.open():
        parts = line.split()
        if len(parts) >= 31:
            boards.append([int(x) for x in parts[:26]])
            probs.append([float(x) for x in parts[26:31]])
    return boards, np.array(probs, dtype=np.float32)


def equity(p: np.ndarray) -> np.ndarray:
    return 2 * p[:, 0] - 1 + p[:, 1] - p[:, 3] + p[:, 2] - p[:, 4]


def er(boards, eq, weights_path: str, n_hidden: int) -> float:
    nn = bgbot_cpp.NNStrategy(weights_path, n_hidden, N_INPUTS)
    e = np.array([nn.evaluate_board(b, b)["equity"] for b in boards])
    return float(np.abs(e - eq).mean() * 1000.0)


def target_noise(meta: Path) -> float | None:
    if not meta.exists():
        return None
    se = [json.loads(line)["se"] for line in meta.open() if line.strip()]
    return float(np.mean(se) * 1000.0 * math.sqrt(2 / math.pi)) if se else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", default="purerace-sage-train-rollout")
    ap.add_argument("--holdout", default="purerace-sage-benchmark-rollout")
    ap.add_argument("--epoch-scale", type=float, default=1.0)
    args = ap.parse_args()

    bgbot_cpp.init_escape_tables()
    tb, tp = load_rollout(_DATA / args.train)
    hb, hp = load_rollout(_DATA / args.holdout)
    heq = equity(hp)
    noise = target_noise(_DATA / (args.holdout + ".meta.jsonl"))
    _OUT.mkdir(parents=True, exist_ok=True)
    wpath = str(_OUT / f"sl_purerace_{args.tag}.weights")
    best = wpath + ".best"
    shutil.copy2(args.init_from, wpath)
    shutil.copy2(args.init_from, best)
    best_er = er(hb, heq, best, args.hidden)
    print(f"=== pure-race SL [{args.tag}]: {len(tb)} train / {len(hb)} holdout rows, "
          f"{args.hidden}h, init {Path(args.init_from).name}: holdout ER {best_er:.2f}"
          + (f" (target noise floor ~{noise:.2f})" if noise else "") + " ===", flush=True)

    inputs = bgbot_cpp.encode_boards_batch(np.array(tb, dtype=np.int32), N_INPUTS)
    t0, total = time.time(), 0
    for phase, (n_epochs, alpha) in enumerate(PHASES, 1):
        n_epochs = max(CHUNK, int(n_epochs * args.epoch_scale))
        shutil.copy2(best, wpath)
        print(f"--- phase {phase}: {n_epochs} ep @ alpha={alpha} (from best {best_er:.2f}) ---",
              flush=True)
        done = 0
        while done < n_epochs:
            chunk = min(CHUNK, n_epochs - done)
            done += chunk
            total += chunk
            sl_train.supervised_train_preencoded(
                inputs=inputs, targets=tp, weights_path=wpath, n_hidden=args.hidden,
                n_inputs=N_INPUTS, alpha=alpha, epochs=chunk, batch_size=BATCH,
                seed=4242 + total, print_interval=chunk + 1, save_path=wpath)
            e = er(hb, heq, wpath, args.hidden)
            mark = ""
            if e < best_er:
                best_er = e
                shutil.copy2(wpath, best)
                mark = "  *BEST*"
            print(f"  P{phase} {done:5d}/{n_epochs}  ER={e:.2f}  best={best_er:.2f}  "
                  f"{time.time() - t0:.0f}s{mark}", flush=True)
    print(f"=== [{args.tag}] done: best holdout ER {best_er:.2f} -> "
          f"{Path(best).relative_to(_PROJECT_ROOT)} ===", flush=True)


if __name__ == "__main__":
    main()
