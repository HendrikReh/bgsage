# SPDX-License-Identifier: MPL-2.0
"""Cubeful 3-ply checker play skips the cubeless filter-and-search pass.

At depths outside CUBEFUL_FILTERED_PLIES every legal move goes through the
cube-aware tree, which replaces each candidate's probs, both equities and
depth label, and the list is re-sorted by a total order. The cubeless pass
that used to run first (1-ply scoring, filter, N-ply search, promotion,
cache clear) therefore decided nothing, and the wrapper now enumerates the
legal boards instead. These tests hold the result to the old path's, field
for field, over every kind of position the two could plausibly differ on.
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

from bgsage import BgBotAnalyzer, STARTING_BOARD, flip_board, possible_moves  # noqa: E402
from bgsage.analyzer import _CubefulAnalyzer, _MultiPlyAnalyzer  # noqa: E402
from bgsage.board import check_game_over  # noqa: E402

# A snake: the 24-NN layout pins every contact node of its tree to NN 22.
SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
# Both sides home: candidates and leaves come from the bearoff database.
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
# On the bar against a closed board: the one candidate is the unchanged board.
BLOCKED = [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -3, 6, 0, 0, 0, 0, 0,
           -2, -2, -2, -2, -2, -2, 1]

CUBES = [
    dict(cube_value=1, cube_owner="centered"),
    dict(cube_value=2, cube_owner="opponent"),
    dict(cube_value=1, cube_owner="centered", away1=5, away2=3),
    dict(cube_value=1, cube_owner="centered", away1=1, away2=4, is_crawford=True),
    dict(cube_value=2, cube_owner="player", max_cube_value=2),
]


def _random_positions(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = list(STARTING_BOARD)
        for _ in range(rng.randint(4, 40)):
            moves = possible_moves(board, rng.randint(1, 6), rng.randint(1, 6))
            if moves:
                board = list(rng.choice(moves))
            if check_game_over(board):
                break
            board = list(flip_board(board))
        else:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            if len(possible_moves(board, d1, d2)) >= 2:
                out.append((board, d1, d2))
    return out


CASES = (
    [(b, d1, d2, CUBES[i % len(CUBES)])
     for i, (b, d1, d2) in enumerate(_random_positions(10, seed=11))]
    + [(SNAKE, 6, 5, CUBES[0]), (SNAKE, 2, 1, CUBES[2]),
       (BEAROFF, 4, 2, CUBES[0]), (BEAROFF, 6, 1, CUBES[3]),
       (BLOCKED, 3, 4, CUBES[0]), (BLOCKED, 6, 6, CUBES[4])]
)


@pytest.fixture(scope="module")
def cf3():
    return BgBotAnalyzer(eval_level="3ply", cubeful=True)


def _moves(analyzer, board, d1, d2, kw, old_path, **extra):
    if old_path:
        with patch.object(_CubefulAnalyzer, "_searches_every_move", return_value=False):
            result = analyzer.checker_play(board, d1, d2, **kw, **extra)
    else:
        result = analyzer.checker_play(board, d1, d2, **kw, **extra)
    return [asdict(m) for m in result.moves]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_three_ply_result_is_the_old_paths(cf3, case):
    board, d1, d2, kw = CASES[case]
    new = _moves(cf3, board, d1, d2, kw, old_path=False)
    old = _moves(cf3, board, d1, d2, kw, old_path=True)
    assert new, "fixture: no legal moves"
    assert new == old
    assert all(m["eval_level"] == "3-ply" for m in new)


def test_force_boards_change_nothing_at_full_width(cf3):
    board, d1, d2, kw = CASES[0]
    worst = _moves(cf3, board, d1, d2, kw, old_path=False)[-1]["board"]
    assert (_moves(cf3, board, d1, d2, kw, old_path=False, force_boards=[worst])
            == _moves(cf3, board, d1, d2, kw, old_path=True, force_boards=[worst]))


def test_the_cubeless_pass_runs_only_where_it_decides_something():
    board, d1, d2, kw = CASES[1]
    for level, runs in (("2ply", True), ("3ply", False)):
        analyzer = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=2)
        inner = analyzer._analyzer._inner
        assert isinstance(inner, _MultiPlyAnalyzer)
        with patch.object(inner, "checker_play_analytics",
                          wraps=inner.checker_play_analytics) as spy, \
                patch.object(bgbot_cpp, "possible_moves",
                             wraps=bgbot_cpp.possible_moves) as moves:
            analyzer.checker_play(board, d1, d2, **kw)
        assert spy.called == runs, level
        assert moves.called
