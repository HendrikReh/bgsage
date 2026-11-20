# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Tests for neongammon.sl_train (the Apple MLX supervised-learning trainer).

These tests run the MLX trainer for an epoch or two on small random data and compare the resulting weights with a
float64 NumPy implementation of the same update rules, using the same shuffle.
"""

import os
import sys

import numpy as np
import pytest

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, 'build'))
sys.path.insert(0, os.path.join(project_dir, 'python'))

import bgbot_cpp  # noqa: E402
from neongammon import sl_train  # noqa: E402

needs_mlx = pytest.mark.skipif(not sl_train.mlx_available(), reason="MLX not available")
WEIGHTS = os.path.join(project_dir, 'models', 'sl_s9_race_race.weights.best')


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def reference_train(params, x, y, sw, alpha, epochs, batch_size, seed):
    """float64 transcription of the trainer's epoch loop."""
    w_h, b_h, w_o, b_o = (np.asarray(p, dtype=np.float64).copy() for p in params)
    rng = np.random.default_rng(seed)
    n = len(x)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x[idx].astype(np.float64), y[idx].astype(np.float64)
            h = _sigmoid(xb @ w_h.T + b_h)
            o = _sigmoid(h @ w_o.T + b_o)
            d_o = (yb - o) * o * (1 - o)
            if sw is not None:
                d_o *= sw[idx][:, None]
            bp = d_o @ w_o
            d_h = bp * h * (1 - h)
            lr = alpha / len(idx)
            w_o += lr * d_o.T @ h
            b_o += lr * d_o.sum(0)
            w_h += lr * d_h.T @ xb
            b_h += lr * d_h.sum(0)
    return w_h, b_h, w_o, b_o


def _data(n, ni, seed=3):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, ni)) < 0.23).astype(np.float32)
    y = rng.random((n, 5)).astype(np.float32)
    sw = rng.uniform(0.5, 4.0, n).astype(np.float32)
    return x, y, sw


def test_weights_file_roundtrip(tmp_path):
    out = tmp_path / 'copy.weights'
    sl_train.write_weights(str(out), *sl_train.read_weights(WEIGHTS))
    assert out.read_bytes() == open(WEIGHTS, 'rb').read()


def test_random_init_matches_neural_network(tmp_path):
    ref = tmp_path / 'ref.weights'
    bgbot_cpp.NeuralNetwork(40, 196, 0.1, 7).save_weights(str(ref))
    for a, b in zip(sl_train._initial_weights('', 40, 196, 7), sl_train.read_weights(str(ref))):
        np.testing.assert_array_equal(a, b)


@needs_mlx
@pytest.mark.parametrize('use_sw', [False, True])
def test_mlx_matches_reference_update(tmp_path, use_sw):
    # 300 rows at batch 128: two full batches and a partial one per epoch.
    x, y, sw = _data(300, 244)
    sw = sw if use_sw else None
    start = sl_train.read_weights(WEIGHTS)
    save = str(tmp_path / 'net.weights')
    r = sl_train._mlx_supervised_train(
        x, y, WEIGHTS, 400, 244, alpha=3.0, epochs=2, batch_size=128, seed=11,
        print_interval=10, save_path=save, benchmark_scenarios=None,
        sample_weights=sw, label='t')
    assert r['epochs_completed'] == 2 and r['best_epoch'] == -1
    got = sl_train.read_weights(save)
    want = reference_train(start, x, y, sw, 3.0, 2, 128, 11)
    for g, w, s in zip(got, want, start):
        assert np.abs(g - s).max() > 1e-4          # the step really moved the weights
        np.testing.assert_allclose(g, w, rtol=0, atol=2e-5)


@needs_mlx
def test_random_init_training_runs(tmp_path):
    x, y, _ = _data(200, 196)
    save = str(tmp_path / 'pr.weights')
    r = sl_train._mlx_supervised_train(
        x, y, '', 40, 196, 1.0, 1, 64, 5, 1, save, None, None, '')
    assert r['epochs_completed'] == 1
    want = reference_train(sl_train._initial_weights('', 40, 196, 5), x, y, None, 1.0, 1, 64, 5)
    for g, w in zip(sl_train.read_weights(save), want):
        np.testing.assert_allclose(g, w, rtol=0, atol=2e-5)


def test_backend_reports_a_trainer():
    assert sl_train.backend() in ('mlx', None)
    assert sl_train.gpu_available() == (sl_train.backend() is not None)
