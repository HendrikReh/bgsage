# SPDX-License-Identifier: MPL-2.0
"""Auto mode of the cheap 1-ply batches starts workers only for enough work.

A 1-ply pre-roll or post-move position costs about 5 us, less than starting
and joining a worker, so one worker per core made small batches slower than
serial. Auto mode now starts one worker per CHEAP_POSITIONS_PER_WORKER
positions there; explicit counts, N-ply levels and checker play are left
alone, and results never depend on the count.
"""

import os
import random
import sys
from dataclasses import asdict
from unittest.mock import patch

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bgbot_cpp  # noqa: E402

from neongammon import (  # noqa: E402
    BatchModels, STARTING_BOARD, batch_evaluate, batch_post_move_evaluate, flip_board,
    possible_moves,
)
from neongammon.batch import CHEAP_POSITIONS_PER_WORKER, _auto_threads  # noqa: E402
from neongammon.board import check_game_over  # noqa: E402


def _rows(n, seed=1429):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = list(STARTING_BOARD)
        for _ in range(rng.randint(4, 40)):
            m = possible_moves(b, rng.randint(1, 6), rng.randint(1, 6))
            if m:
                b = list(rng.choice(m))
            if check_game_over(b):
                break
            b = list(flip_board(b))
        else:
            out.append({"board": b, "cube_value": 1, "cube_owner": "centered"})
    return out


def test_auto_thread_rule():
    m = CHEAP_POSITIONS_PER_WORKER
    with patch("neongammon.batch.os.cpu_count", lambda: 14):
        assert [_auto_threads(0, n, 1) for n in (0, 1, m - 1, m, 2 * m, 5 * m, 1000)] == \
            [1, 1, 1, 1, 2, 5, 14]
        assert _auto_threads(3, 1000, 1) == 3          # explicit counts as given
        assert _auto_threads(0, 4, 2) == 0             # N-ply: native default


@pytest.mark.parametrize("fn,native", [
    (batch_evaluate, "batch_evaluate_positions"),
    (batch_post_move_evaluate, "batch_evaluate_post_move"),
])
def test_small_auto_batches_run_on_one_worker(fn, native):
    call = getattr(bgbot_cpp, native)
    with patch.object(bgbot_cpp, native, side_effect=call) as spy:
        fn(_rows(4), n_threads=0)
    assert spy.call_args.args[2] == 1


@pytest.mark.parametrize("level", ["1ply", "2ply"])
@pytest.mark.parametrize("fn", [batch_evaluate, batch_post_move_evaluate])
def test_results_do_not_depend_on_thread_count(fn, level):
    rows, models = _rows(40), BatchModels()
    out = [[asdict(r) for r in fn(rows, eval_level=level, n_threads=n, models=models)]
           for n in (1, 2, 4, 0)]
    assert all(o == out[0] for o in out)
