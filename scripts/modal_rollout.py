# SPDX-License-Identifier: MPL-2.0
"""
Roll out post-move positions on Modal (https://modal.com) CPU containers.

Input is a position file in the repo's training-data format: one post-move
board per line, 26 space-separated ints, the side that just moved positive
(what the harvest scripts write, e.g. ``data/s11-bg-containment-data``).
Output is the matching rollout file: the same 26 ints followed by the five
cubeless rolled-out probabilities to 4 decimals (W, Gw, Bw, Gl, Bl, from the
positive side's perspective) — the format ``run_s11_containment_sl.py`` and
``run_s11_backgame_sl.py`` read. A ``.meta.jsonl`` sidecar keeps equity,
standard errors and per-board seconds.

Default rollout convention (the one the S11 piles were rolled out with):
Stage 9 play, 1,296 trials, no truncation, 3-ply checker play throughout
(``ultra_late_threshold=9999``), VR on, PubEval 20/15 prefilter, bearoff DB.
Every knob is a flag.

Fully resumable: boards already in the output file are skipped, results are
appended as they arrive, so an interrupted run loses only the boards in
flight. ``--max-usd`` stops the run once the estimated compute cost of this
run reaches the limit (the estimate counts rollout time only — container
start-up and idle time are extra; Modal's dashboard has the real bill).

The engine is built inside the Modal image from ``cpp/`` with the inherited
Linux x86-64 flags (AVX2/FMA, -ffast-math). Its values differ from the Apple
Accelerate build at the ~1e-3 equity level, well under a rollout's standard
error (~0.009 at 1,296 trials).

Usage (from the repo root, with ``pip install modal`` and ``modal setup`` done):

    # local sanity run on this Mac, same code path, no Modal
    python scripts/modal_rollout.py data/s11-bg-containment-data --local --limit 2

    # small cloud test: 20 boards, report measured cost per board
    python scripts/modal_rollout.py data/s11-bg-containment-data --limit 20 \\
        --out logs/modal_test-rollout

    # real run: 100 containers x 8 cores, stop at $300
    python scripts/modal_rollout.py data/s11-bg-snake-exit1-data \\
        --n-trials 648 --containers 100 --max-usd 300
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import modal

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Modal list prices (modal.com/pricing, checked 2026-11-18): CPU is billed per
# PHYSICAL core (2 vCPU) per second, memory per GiB per second, each on
# max(request, usage).
USD_PER_CORE_SEC = 0.0000131
USD_PER_GIB_SEC = 0.00000222

REMOTE_ROOT = "/root/bgsage"

# --------------------------------------------------------------------------
# Engine-side code (runs locally with --local, and inside the containers)
# --------------------------------------------------------------------------


def _engine_paths(root: str) -> None:
    for p in (os.path.join(root, "python"), os.path.join(root, "build")):
        if p not in sys.path:
            sys.path.insert(0, p)


def make_rollout(root: str, model: str, n_trials: int, checker_ply: int,
                 truncation_depth: int, n_threads: int, seed: int):
    """Build the rollout strategy. Returns (strategy, keepalive objects)."""
    _engine_paths(root)
    import bgbot_cpp
    from bgsage.weights import WeightConfig, bearoff_db_path

    bgbot_cpp.init_escape_tables()
    w = WeightConfig.from_model(model)
    w.validate()
    strat = bgbot_cpp.create_rollout(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
        n_trials=n_trials,
        truncation_depth=truncation_depth,
        decision_ply=checker_ply,
        n_threads=n_threads,
        seed=seed,
        enable_vr=True,
        ultra_late_threshold=9999,
    )
    pubeval = bgbot_cpp.PubEval()
    strat.set_move_filter(pubeval)
    keep = [pubeval]
    db_path = bearoff_db_path()
    if db_path:
        db = bgbot_cpp.BearoffDB()
        if db.load(db_path):
            bgbot_cpp.rollout_set_bearoff_db(strat, db)
            keep.append(db)
    return strat, keep


def rollout_one(strat, board: list[int]) -> dict:
    t0 = time.perf_counter()
    r = strat.rollout_position(list(board))
    return {
        "board": list(board),
        "probs": [float(p) for p in r.mean_probs],
        "equity": float(r.equity),
        "se": float(r.std_error),
        "prob_se": [float(p) for p in r.prob_std_errors],
        "secs": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------
# Modal app
# --------------------------------------------------------------------------

def _weights_ignore(p: Path) -> bool:
    # Upload only the registry's weight files, not TD checkpoints or the
    # diagnostic subfolders.
    return not (p.name.startswith("sl_") and p.name.endswith(".weights.best"))


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "cmake", "ninja-build")
    .pip_install("numpy", "pybind11")
    .add_local_dir(_PROJECT_ROOT / "cpp", f"{REMOTE_ROOT}/cpp", copy=True,
                   ignore=["move_array_probe"])
    .run_commands(
        # cpp/CMakeLists.txt is the upstream CUDA build; inside the image the
        # CPU list file takes its place (it resolves sources relative to itself).
        f"cp {REMOTE_ROOT}/cpp/CMakeLists_cpu.txt {REMOTE_ROOT}/cpp/CMakeLists.txt",
        f"mkdir -p {REMOTE_ROOT}/build",
        f"cmake -S {REMOTE_ROOT}/cpp -B /tmp/bgbuild -G Ninja -DCMAKE_BUILD_TYPE=Release"
        " -DBGBOT_BUILD_APPS=OFF -Dpybind11_DIR=$(python -m pybind11 --cmakedir)",
        "ninja -C /tmp/bgbuild bgbot_cpp",
        f"cp /tmp/bgbuild/bgbot_cpp*.so {REMOTE_ROOT}/build/",
    )
    .add_local_dir(_PROJECT_ROOT / "python" / "bgsage", f"{REMOTE_ROOT}/python/bgsage",
                   ignore=["**/__pycache__"])
    .add_local_dir(_PROJECT_ROOT / "models", f"{REMOTE_ROOT}/models",
                   ignore=_weights_ignore)
    .add_local_file(_PROJECT_ROOT / "data" / "bearoff_1sided.db",
                    f"{REMOTE_ROOT}/data/bearoff_1sided.db")
)

app = modal.App("bgsage-rollout", image=image)


@app.cls(cpu=8.0, memory=4096, timeout=3 * 3600, retries=1, max_containers=50)
class Roller:
    model: str = modal.parameter(default="stage9")
    n_trials: int = modal.parameter(default=1296)
    checker_ply: int = modal.parameter(default=3)
    truncation_depth: int = modal.parameter(default=0)
    threads: int = modal.parameter(default=16)
    seed: int = modal.parameter(default=42)

    @modal.enter()
    def load(self):
        self.strat, self._keep = make_rollout(
            REMOTE_ROOT, self.model, self.n_trials, self.checker_ply,
            self.truncation_depth, self.threads, self.seed)

    @modal.method()
    def rollout(self, board: list[int]) -> dict:
        return rollout_one(self.strat, board)

    @modal.method()
    def rollout_many(self, boards: list[list[int]]) -> list[dict]:
        # One input per chunk: for sub-second boards (races) the per-input
        # overhead would otherwise rival the rollout itself.
        return [rollout_one(self.strat, b) for b in boards]


# --------------------------------------------------------------------------
# Local driver
# --------------------------------------------------------------------------


def _load_boards(path: Path) -> list[list[int]]:
    boards = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 26:
                continue
            boards.append([int(x) for x in parts[:26]])
    return boards


def _done_keys(path: Path) -> set[tuple[int, ...]]:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 31:
                    done.add(tuple(int(x) for x in parts[:26]))
    return done


def _default_out(positions: Path) -> Path:
    name = positions.name
    if name.endswith("-data"):
        name = name[: -len("-data")]
    return positions.with_name(name + "-rollout")


def _format_row(res: dict) -> str:
    return (" ".join(str(x) for x in res["board"]) + " "
            + " ".join(f"{p:.4f}" for p in res["probs"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("positions", type=Path, help="position file (26 ints per line)")
    ap.add_argument("--out", type=Path, default=None,
                    help="rollout file to append to (default: <name>-rollout next to the input)")
    ap.add_argument("--model", default="stage9", help="model that plays the trials (default stage9)")
    ap.add_argument("--n-trials", type=int, default=1296)
    ap.add_argument("--checker-ply", type=int, default=3)
    ap.add_argument("--truncation-depth", type=int, default=0, help="0 = play to the end")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="roll out at most N pending boards")
    ap.add_argument("--cpu", type=float, default=8.0, help="physical cores per container")
    ap.add_argument("--memory", type=int, default=4096, help="MiB per container")
    ap.add_argument("--containers", type=int, default=50, help="max concurrent containers")
    ap.add_argument("--timeout", type=int, default=3 * 3600,
                    help="seconds per input (one board, or one --chunk)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="boards per Modal input; raise it (e.g. 50) for sub-second boards")
    ap.add_argument("--max-usd", type=float, default=0.0,
                    help="stop once this run's estimated compute cost reaches it (0 = no limit)")
    ap.add_argument("--local", action="store_true", help="run in-process on this machine instead")
    ap.add_argument("--threads", type=int, default=0,
                    help="rollout threads (default: 2 x --cpu on Modal, all cores with --local)")
    args = ap.parse_args()

    positions = args.positions if args.positions.is_absolute() else Path.cwd() / args.positions
    out = args.out or _default_out(positions)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = out.with_name(out.name + ".meta.jsonl")

    done = _done_keys(out)
    seen: set[tuple[int, ...]] = set()
    pending = []
    for b in _load_boards(positions):
        k = tuple(b)
        if k in done or k in seen:
            continue
        seen.add(k)
        pending.append(b)
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(done)} boards already in {out}; {len(pending)} to roll out", flush=True)
    if not pending:
        return

    config = dict(model=args.model, n_trials=args.n_trials, checker_ply=args.checker_ply,
                  truncation_depth=args.truncation_depth, seed=args.seed)
    usd_per_sec = args.cpu * USD_PER_CORE_SEC + (args.memory / 1024) * USD_PER_GIB_SEC

    n_ok = n_fail = 0
    busy_secs = 0.0
    t_start = time.time()

    def record(res: dict, fo, fm) -> None:
        nonlocal n_ok, busy_secs
        fo.write(_format_row(res) + "\n")
        fo.flush()
        fm.write(json.dumps({k: res[k] for k in ("board", "equity", "se", "prob_se", "secs")}
                            | config) + "\n")
        fm.flush()
        n_ok += 1
        busy_secs += res["secs"]

    def progress() -> str:
        wall = time.time() - t_start
        est = busy_secs * usd_per_sec
        per = est / n_ok if n_ok else 0.0
        return (f"{n_ok}/{len(pending)} done, {n_fail} failed, {wall / 60:.1f} min wall, "
                f"{busy_secs / max(n_ok, 1):.1f} s/board, est ${est:.2f} "
                f"(${per:.4f}/board)")

    with open(out, "a") as fo, open(meta, "a") as fm:
        if args.local:
            threads = args.threads or os.cpu_count() or 1
            strat, _keep = make_rollout(
                str(_PROJECT_ROOT), args.model, args.n_trials, args.checker_ply,
                args.truncation_depth, threads, args.seed)
            for b in pending:
                record(rollout_one(strat, b), fo, fm)
                print(progress(), flush=True)
            return

        # A plain SIGTERM would kill the client without leaving app.run(), and
        # Modal keeps the ephemeral app's containers running (and billing)
        # until it notices; turn it into a KeyboardInterrupt so the app stops.
        def _term(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)

        threads = args.threads or int(round(2 * args.cpu))
        roller_cls = Roller.with_options(cpu=args.cpu, memory=args.memory,
                                         timeout=args.timeout, max_containers=args.containers)
        with modal.enable_output(), app.run():
            roller = roller_cls(threads=threads, **config)
            last = 0.0
            if args.chunk > 1:
                chunks = [pending[i:i + args.chunk] for i in range(0, len(pending), args.chunk)]
                stream = roller.rollout_many.map(chunks, order_outputs=False,
                                                 return_exceptions=True)
            else:
                stream = roller.rollout.map(pending, order_outputs=False,
                                            return_exceptions=True)
            for item in stream:
                if isinstance(item, BaseException):
                    n_fail += args.chunk
                    print(f"input failed ({args.chunk} board(s)): {item!r}", flush=True)
                    continue
                for res in (item if args.chunk > 1 else [item]):
                    record(res, fo, fm)
                if time.time() - last > 30 or n_ok == len(pending):
                    print(progress(), flush=True)
                    last = time.time()
                if args.max_usd and busy_secs * usd_per_sec >= args.max_usd:
                    print(f"--max-usd {args.max_usd} reached; stopping (re-run to resume)",
                          flush=True)
                    break
    print(progress(), flush=True)


if __name__ == "__main__":
    main()
