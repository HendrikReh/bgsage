# SPDX-License-Identifier: MPL-2.0
"""Cubeful 2-ply checker play searches only the 1-ply filter's survivors.

The cube-aware 2-ply tree used to score every legal move, so the pick could
fall on a move the filter rejected (snake folder, 2026-11-19: 103 of 795
cubeful 2-ply picks, 2.2-2.9 PR worse under XG and both rollout references).
Now the pruned candidates keep 1-ply values, as in the cubeless path, and
only enter the search through the stale-top-two promotion. 3-ply keeps
scoring every move (CUBEFUL_FILTERED_PLIES). batch_checker_play follows the
same steps (tests/test_batch_checker_play_parity.py).
"""

import os
import random
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neongammon import BgBotAnalyzer, STARTING_BOARD, flip_board, possible_moves  # noqa: E402
from neongammon.board import check_game_over  # noqa: E402


def _positions(n=24, seed=7):
    """Mid-game positions from random play, with the roll to play."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = list(STARTING_BOARD)
        for _ in range(rng.randint(6, 30)):
            moves = possible_moves(board, rng.randint(1, 6), rng.randint(1, 6))
            if moves:
                board = list(rng.choice(moves))
            if check_game_over(board):
                break
            board = list(flip_board(board))
        else:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            if len(possible_moves(board, d1, d2)) >= 8:
                out.append((board, d1, d2))
    return out


@pytest.fixture(scope="module")
def analyzers():
    return {
        "cf2": BgBotAnalyzer(eval_level="2ply", cubeful=True),
        "cl2": BgBotAnalyzer(eval_level="2ply", cubeful=False),
        "cf1": BgBotAnalyzer(eval_level="1ply", cubeful=True),
    }


@pytest.mark.parametrize("cube_value,cube_owner", [(1, "centered"), (2, "player")])
def test_pruned_moves_stay_1ply_and_never_lead(analyzers, cube_value, cube_owner):
    n_pruned = 0
    for board, d1, d2 in _positions():
        kw = dict(cube_value=cube_value, cube_owner=cube_owner)
        cf = analyzers["cf2"].checker_play(board, d1, d2, **kw).moves
        cl = analyzers["cl2"].checker_play(board, d1, d2, **kw).moves
        one = {tuple(m.board): m for m in analyzers["cf1"].checker_play(board, d1, d2, **kw).moves}
        cl_pruned = {tuple(m.board) for m in cl if m.eval_level == "1-ply"}
        cf_pruned = {tuple(m.board) for m in cf if m.eval_level == "1-ply"}
        # Only moves the cubeless filter pruned can be 1-ply here; a few of
        # them may have been promoted into the search, none the other way.
        assert cf_pruned <= cl_pruned
        # Nothing in the top two carries a 1-ply value.
        assert all(m.eval_level == "2-ply" for m in cf[:2])
        # A pruned move carries exactly the 1-ply analyzer's cubeful value.
        for m in cf:
            if m.eval_level == "1-ply":
                ref = one[tuple(m.board)]
                assert m.equity == pytest.approx(ref.equity, abs=1e-9)
                assert m.cubeless_equity == pytest.approx(ref.cubeless_equity, abs=1e-9)
        n_pruned += len(cf_pruned)
    assert n_pruned > 0, "no position exercised the filter"


def test_best_move_comes_from_the_search(analyzers):
    """The cubeful pick is a move the cubeless N-ply search also searched
    (a survivor or promoted there) or one this path promoted itself."""
    for board, d1, d2 in _positions():
        cf = analyzers["cf2"].checker_play(board, d1, d2).moves
        cl = analyzers["cl2"].checker_play(board, d1, d2).moves
        searched = {tuple(m.board) for m in cl if m.eval_level != "1-ply"}
        cf_searched = {tuple(m.board) for m in cf if m.eval_level != "1-ply"}
        assert tuple(cf[0].board) in cf_searched
        # Every cubeless survivor is searched here too.
        assert searched <= cf_searched


def test_3ply_still_scores_every_move():
    """3-ply is not in CUBEFUL_FILTERED_PLIES: there the full search pays
    (money benchmark, 2026-11-19), so every candidate stays 3-ply."""
    from neongammon.analyzer import CUBEFUL_FILTERED_PLIES

    assert 3 not in CUBEFUL_FILTERED_PLIES
    an = BgBotAnalyzer(eval_level="3ply", cubeful=True)
    for board, d1, d2 in _positions(n=3):
        assert all(m.eval_level == "3-ply" for m in an.checker_play(board, d1, d2).moves)
