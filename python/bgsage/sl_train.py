# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""GPU supervised-learning (SL) training of one network, CUDA or Apple MLX.

The training scripts call :func:`supervised_train` and
:func:`supervised_train_preencoded`, which take the same arguments and return
the same dict as ``bgbot_cpp.cuda_supervised_train`` /
``bgbot_cpp.cuda_supervised_train_preencoded``:

* **CUDA** (``bgbot_cpp.cuda_available()``): forwards to the C++ trainer.
* **MLX** (Apple Silicon, ``pip install mlx``): the same algorithm on the Apple
  GPU, implemented here.

:func:`backend` reports which one is used. The MLX trainer reproduces
``cpp/src/cuda_nn.cu`` step for step:

* N_inputs -> N_hidden (sigmoid) -> 5 (sigmoid); sigmoid-MSE output delta
  ``(t - p) * p * (1 - p)``, scaled by the per-sample weight when given, and
  backpropagated through the output weights as they were before the step.
* Plain SGD: every weight and bias moves by ``alpha / batch_rows`` times its
  summed batch gradient (the final, partial batch uses its own row count).
* One epoch is one pass over the shuffled data; when benchmark scenarios are
  given, the net is scored on the CPU after every epoch and the best-scoring
  weights are written to ``save_path + ".best"``; ``save_path`` is written every
  ``print_interval`` epochs and after the last one.
* Weights start from ``weights_path``, or from ``bgbot_cpp.NeuralNetwork``'s
  random init with the same ``seed`` (the identical initial weights CUDA gets).

Not bit-identical to CUDA: the shuffle order comes from NumPy rather than
``std::mt19937``, MLX's sigmoid does not clamp at +-15 the way the CUDA kernel
does (a difference below 3e-7), and float summation order differs between GPUs.
"""

from __future__ import annotations

import os
import struct
import tempfile
import time

import numpy as np

import bgbot_cpp

NUM_OUTPUTS = 5
_EVAL_EVERY = 16  # MLX steps between forced evaluations (bounds the lazy graph)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def mlx_available() -> bool:
    """True when MLX can be imported and has a GPU device."""
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:  # ImportError, or an MLX build without Metal
        return False


def backend() -> str | None:
    """``"cuda"``, ``"mlx"`` or ``None`` (no GPU trainer available)."""
    if bgbot_cpp.cuda_available():
        return "cuda"
    if mlx_available():
        return "mlx"
    return None


def gpu_available() -> bool:
    """True when either GPU trainer can run (the scripts' old cuda_available() gate)."""
    return backend() is not None


# ---------------------------------------------------------------------------
# Weight files (NeuralNetwork::save_weights format)
# ---------------------------------------------------------------------------

def read_weights(path: str):
    """Read a ``.weights`` file into (W_h [nh, ni], b_h [nh], W_o [5, nh], b_o [5])."""
    with open(path, "rb") as f:
        nh, ni, no = struct.unpack("<3i", f.read(12))
        hidden = np.frombuffer(f.read(nh * (ni + 1) * 4), dtype="<f4").reshape(nh, ni + 1)
        output = np.frombuffer(f.read(no * (nh + 1) * 4), dtype="<f4").reshape(no, nh + 1)
    return (hidden[:, :ni].copy(), hidden[:, ni].copy(),
            output[:, :nh].copy(), output[:, nh].copy())


def write_weights(path: str, w_h, b_h, w_o, b_o) -> None:
    """Write weights in NeuralNetwork::save_weights format (bias in the last column)."""
    w_h, b_h, w_o, b_o = (np.asarray(a, dtype=np.float32) for a in (w_h, b_h, w_o, b_o))
    nh, ni = w_h.shape
    hidden = np.concatenate([w_h, b_h[:, None]], axis=1)
    output = np.concatenate([w_o, b_o[:, None]], axis=1)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<3i", nh, ni, w_o.shape[0]))
        f.write(hidden.astype("<f4").tobytes())
        f.write(output.astype("<f4").tobytes())
    os.replace(tmp, path)


def _initial_weights(weights_path: str, n_hidden: int, n_inputs: int, seed: int):
    if weights_path:
        w = read_weights(weights_path)
        if w[0].shape != (n_hidden, n_inputs):
            raise RuntimeError(
                f"{weights_path} is {w[0].shape[0]}h x {w[0].shape[1]} inputs, "
                f"expected {n_hidden}h x {n_inputs}")
        return w
    # Same random init as the CUDA trainer: NeuralNetwork(nh, ni, eps=0.1, seed).
    nn = bgbot_cpp.NeuralNetwork(n_hidden, n_inputs, 0.1, seed)
    fd, tmp = tempfile.mkstemp(suffix=".weights")
    os.close(fd)
    try:
        if not nn.save_weights(tmp):
            raise RuntimeError("could not write the initial random weights")
        return read_weights(tmp)
    finally:
        os.remove(tmp)


# ---------------------------------------------------------------------------
# MLX trainer
# ---------------------------------------------------------------------------

def _make_step(mx, with_sample_weights: bool):
    def step(w_h, b_h, w_o, b_o, x, y, sw, lr):
        h = mx.sigmoid(x @ w_h.T + b_h)
        o = mx.sigmoid(h @ w_o.T + b_o)
        d_o = (y - o) * o * (1.0 - o)
        if with_sample_weights:
            d_o = d_o * sw[:, None]
        bp = d_o @ w_o                      # through the pre-update output weights
        d_h = bp * h * (1.0 - h)
        return (w_h + lr * (d_h.T @ x), b_h + lr * d_h.sum(axis=0),
                w_o + lr * (d_o.T @ h), b_o + lr * d_o.sum(axis=0))
    return mx.compile(step)


def _mlx_supervised_train(inputs, targets, weights_path, n_hidden, n_inputs, alpha,
                          epochs, batch_size, seed, print_interval, save_path,
                          benchmark_scenarios, sample_weights, label):
    import mlx.core as mx

    t_start = time.perf_counter()
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)
    n_pos = inputs.shape[0]
    if inputs.ndim != 2 or inputs.shape[1] != n_inputs:
        raise RuntimeError(f"inputs must be [N, {n_inputs}], got {inputs.shape}")
    if targets.shape != (n_pos, NUM_OUTPUTS):
        raise RuntimeError(f"targets must be [{n_pos}, 5], got {targets.shape}")
    if sample_weights is not None:
        sample_weights = np.ascontiguousarray(sample_weights, dtype=np.float32)
        if sample_weights.shape != (n_pos,):
            raise RuntimeError(f"sample_weights must be [{n_pos}], got {sample_weights.shape}")
    print_interval = max(1, int(print_interval))

    x_all = mx.array(inputs)
    y_all = mx.array(targets)
    sw_all = mx.array(sample_weights) if sample_weights is not None else None
    params = [mx.array(a) for a in _initial_weights(weights_path, n_hidden, n_inputs, seed)]
    step = _make_step(mx, sw_all is not None)
    dummy_sw = mx.zeros((1,), dtype=mx.float32)

    scenarios = benchmark_scenarios if benchmark_scenarios is not None and len(benchmark_scenarios) else None
    best_path = f"{save_path}.best" if save_path else ""
    result = {"best_score": 1e9, "best_epoch": -1, "epochs_completed": 0,
              "total_seconds": 0.0, "weights_path": save_path,
              "best_weights_path": best_path}
    scores: list[float] = []
    rng = np.random.default_rng(seed)
    bench_tmp = None
    if scenarios is not None:
        fd, bench_tmp = tempfile.mkstemp(suffix=".weights")
        os.close(fd)

    def host_weights():
        mx.eval(params)
        return [np.array(p) for p in params]

    # Learning rate per batch: alpha / rows, with rows < batch_size only for the
    # final partial batch of each epoch.
    lr_full = mx.array(alpha / batch_size, dtype=mx.float32)
    tail_rows = n_pos % batch_size
    lr_tail = mx.array(alpha / tail_rows, dtype=mx.float32) if tail_rows else lr_full
    n_steps = 0

    try:
        for epoch in range(epochs):
            order = mx.array(rng.permutation(n_pos).astype(np.int32))
            for start in range(0, n_pos, batch_size):
                idx = order[start:start + batch_size]
                x = mx.take(x_all, idx, axis=0)
                y = mx.take(y_all, idx, axis=0)
                sw = mx.take(sw_all, idx, axis=0) if sw_all is not None else dummy_sw
                lr = lr_full if start + batch_size <= n_pos else lr_tail
                params = list(step(*params, x, y, sw, lr))
                n_steps += 1
                if n_steps % _EVAL_EVERY == 0:
                    mx.eval(params)

            score = -1.0
            if scenarios is not None:
                write_weights(bench_tmp, *host_weights())
                score = bgbot_cpp.score_benchmarks_nn(
                    scenarios, bench_tmp, n_hidden, n_inputs, 0).score()
                scores.append(score)

            last = epoch == epochs - 1
            if epoch % print_interval == 0 or last:
                line = (f"Epoch {epoch:4d}  score={score:.2f}  "
                        f"time={time.perf_counter() - t_start:.1f}s")
                if len(scores) >= 10:
                    avg = sum(scores[-10:]) / 10
                    line += f"  avg10={avg:.2f}"
                    if len(scores) >= 20:
                        line += f" ({avg - sum(scores[-20:-10]) / 10:+.2f})"
                print(line, flush=True)

            if score >= 0 and score < result["best_score"]:
                result["best_score"] = score
                result["best_epoch"] = epoch
                if best_path:
                    write_weights(best_path, *host_weights())
            if save_path and (epoch % print_interval == 0 or last):
                write_weights(save_path, *host_weights())
            result["epochs_completed"] = epoch + 1
    finally:
        if bench_tmp and os.path.exists(bench_tmp):
            os.remove(bench_tmp)

    result["total_seconds"] = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# Public entry points (same signatures as the bgbot_cpp CUDA bindings)
# ---------------------------------------------------------------------------

def supervised_train_preencoded(inputs, targets, weights_path="", n_hidden=120,
                                n_inputs=196, alpha=1.0, epochs=100, batch_size=4096,
                                seed=42, print_interval=1, save_path="",
                                benchmark_scenarios=None, sample_weights=None, label=""):
    """SL training on pre-encoded inputs ``[N, n_inputs]`` and targets ``[N, 5]``.

    Same arguments and return dict as ``bgbot_cpp.cuda_supervised_train_preencoded``
    (``best_score``, ``best_epoch``, ``epochs_completed``, ``total_seconds``,
    ``weights_path``, ``best_weights_path``). Uses CUDA when available, else MLX.
    """
    kind = backend()
    if kind == "cuda":
        return bgbot_cpp.cuda_supervised_train_preencoded(
            inputs=inputs, targets=targets, weights_path=weights_path,
            n_hidden=n_hidden, n_inputs=n_inputs, alpha=alpha, epochs=epochs,
            batch_size=batch_size, seed=seed, print_interval=print_interval,
            save_path=save_path, benchmark_scenarios=benchmark_scenarios,
            sample_weights=sample_weights, label=label)
    if kind == "mlx":
        return _mlx_supervised_train(
            inputs, targets, weights_path, n_hidden, n_inputs, alpha, epochs,
            batch_size, seed, print_interval, save_path, benchmark_scenarios,
            sample_weights, label)
    raise RuntimeError("No GPU SL trainer: CUDA is unavailable and MLX is not "
                       "installed (pip install mlx on Apple Silicon)")


def supervised_train(boards, targets, weights_path="", n_hidden=120, n_inputs=196,
                     alpha=1.0, epochs=100, batch_size=4096, seed=42, print_interval=1,
                     save_path="", benchmark_scenarios=None, sample_weights=None, label=""):
    """SL training on boards ``[N, 26]`` (encoded on the CPU first) and targets ``[N, 5]``.

    Same arguments and return dict as ``bgbot_cpp.cuda_supervised_train``.
    """
    kind = backend()
    if kind == "cuda":
        return bgbot_cpp.cuda_supervised_train(
            boards=boards, targets=targets, weights_path=weights_path,
            n_hidden=n_hidden, n_inputs=n_inputs, alpha=alpha, epochs=epochs,
            batch_size=batch_size, seed=seed, print_interval=print_interval,
            save_path=save_path, benchmark_scenarios=benchmark_scenarios,
            sample_weights=sample_weights, label=label)
    boards = np.ascontiguousarray(boards, dtype=np.int32)
    if boards.ndim != 2 or boards.shape[1] != 26:
        raise RuntimeError("boards must be shape [N, 26]")
    inputs = bgbot_cpp.encode_boards_batch(boards, n_inputs)
    msg = (f"Pre-encoded {len(boards)} positions ({n_inputs} inputs each)  hidden={n_hidden}"
           f"  alpha={alpha}  batch={batch_size}  seed={seed}")
    if label:
        msg += f"  [{label}]"
    if sample_weights is not None:
        msg += "  (sample weights enabled)"
    print(msg + f"  [backend: {kind or 'none'}]", flush=True)
    return supervised_train_preencoded(
        inputs, targets, weights_path=weights_path, n_hidden=n_hidden,
        n_inputs=n_inputs, alpha=alpha, epochs=epochs, batch_size=batch_size,
        seed=seed, print_interval=print_interval, save_path=save_path,
        benchmark_scenarios=benchmark_scenarios, sample_weights=sample_weights,
        label=label)
