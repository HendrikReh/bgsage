# SPDX-License-Identifier: MPL-2.0
"""The legacy thread-local PosCache is allocated on first insert.

Only the old cubeless recursion (hybrid evaluator, full-depth opponent)
inserts into it; the default tree goes through the cubeful engine and never
does. The table (~36 MB per thread) used to be allocated and zeroed by every
thread that called clear_cache(), each rollout worker included, and every
clear zeroed all of it. Now it is allocated on the first insert, and a clear
zeroes it only when something was inserted since the last one. These tests
check that the cache still caches, clears and returns the same values.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bgbot_cpp  # noqa: E402

from bgsage import STARTING_BOARD, possible_moves  # noqa: E402
from bgsage.weights import WeightConfig, default_weights  # noqa: E402

CANDIDATES = [list(c) for c in possible_moves(STARTING_BOARD, 3, 1)[:4]]


def _hybrid():
    w = WeightConfig.from_model("stage5")
    return bgbot_cpp.create_multipy_hybrid_5nn(*w.weight_args, *w.weight_args, n_plies=2)


def _probs(strategy):
    return [list(strategy.evaluate_board(c, STARTING_BOARD)["probs"]) for c in CANDIDATES]


def test_the_legacy_recursion_caches_and_clears():
    strategy = _hybrid()
    strategy.clear_cache()
    assert strategy.cache_size() == 0
    first = _probs(strategy)
    assert strategy.cache_size() > 0
    # Served from the cache: same values.
    assert _probs(strategy) == first
    strategy.clear_cache()
    assert strategy.cache_size() == 0
    # Recomputed after the clear: still the same values.
    assert _probs(strategy) == first
    strategy.clear_cache()


def test_the_default_tree_never_fills_it():
    w = default_weights()
    strategy = bgbot_cpp.create_multipy(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list, n_plies=2)
    strategy.clear_cache()
    _probs(strategy)
    assert strategy.cache_size() == 0
    # Clearing an untouched cache is harmless, however often.
    for _ in range(3):
        strategy.clear_cache()
    assert strategy.cache_size() == 0
