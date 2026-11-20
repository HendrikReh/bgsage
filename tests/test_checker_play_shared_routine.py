# SPDX-License-Identifier: MPL-2.0
"""Single and batch N-ply checker play run one native routine.

BgBotAnalyzer.checker_play (cube-aware N-ply) and batch_checker_play used to
implement the 2-ply filter, the cubeless survivor search and promotions, the
cube-aware rescoring and the final top-two promotions twice, in Python and in
C++. Both now go through score_checker_position in the bindings: the analyzer
via bgbot_cpp.checker_play_nply (with its forced boards, cube cap and thread
budget), the batch per position. These tests hold the two entry points to the
same lists, check the single-position options, and check that the old Python
pass no longer runs.
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
from bgsage.analyzer import (  # noqa: E402
    CUBEFUL_FILTERED_PLIES, _contact_cube_life, _contact_cube_slope, resolve_owner,
)
from bgsage.board import check_game_over  # noqa: E402

SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
BLOCKED = [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -3, 6, 0, 0, 0, 0, 0,
           -2, -2, -2, -2, -2, -2, 1]
CUBES = [
    dict(cube_value=1, cube_owner="centered"),
    dict(cube_value=2, cube_owner="opponent"),
    dict(cube_value=1, cube_owner="centered", away1=5, away2=3),
    dict(cube_value=1, cube_owner="centered", away1=1, away2=4, is_crawford=True),
    dict(cube_value=2, cube_owner="player", max_cube_value=2),
    dict(cube_value=2, cube_owner="player", max_cube_value=4),
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
     for i, (b, d1, d2) in enumerate(_random_positions(12, seed=11))]
    + [(SNAKE, 6, 5, CUBES[0]), (SNAKE, 2, 1, CUBES[2]),
       (BEAROFF, 4, 2, CUBES[0]), (BEAROFF, 6, 1, CUBES[3]),
       (BLOCKED, 3, 4, CUBES[0]), (BLOCKED, 6, 6, CUBES[4])]
)


def _single(a, board, d1, d2, kw, **extra):
    return [asdict(m) for m in a.checker_play(board, d1, d2, **kw, **extra).moves]


def _batch(a, cases, force=None):
    """The native batch over the analyzer's own strategies and settings."""
    inner = a._analyzer._inner
    n = inner._n_plies
    inputs = []
    for i, (b, d1, d2, kw) in enumerate(cases):
        d = dict(board=list(b), die1=d1, die2=d2, cube_value=kw.get("cube_value", 1),
                 cube_owner=resolve_owner(kw.get("cube_owner", "centered")))
        if kw.get("away1"):
            d.update(away1=kw["away1"], away2=kw["away2"], is_crawford=kw.get("is_crawford", False))
        if kw.get("max_cube_value"):
            d["max_cube_value"] = kw["max_cube_value"]
        if force:
            d["force_boards"] = force[i]
        inputs.append(d)
    out = []
    for d in inputs:    # one row per call: money and match rows take different cube lives
        a1, a2 = (d.get("away1", 0), d.get("away2", 0))
        r = bgbot_cpp.batch_checker_play(
            [d], inner._strategy_1ply, inner._strategy_nply, inner.FILTER_MAX_MOVES,
            inner.FILTER_THRESHOLD, 1, jacoby=True, beaver=True,
            contact_x_1ply=_contact_cube_life(1, a1, a2),
            contact_x_pip_slope_1ply=_contact_cube_slope(1, a1, a2),
            contact_x_nply=_contact_cube_life(n, a1, a2),
            contact_x_pip_slope_nply=_contact_cube_slope(n, a1, a2),
            cubeful_filtered_plies=sorted(CUBEFUL_FILTERED_PLIES))[0]
        out.append(r["moves"])
    return out


def _as_batch_dicts(moves):
    return [{k: m[k] for k in ("board", "equity", "cubeless_equity", "equity_diff", "eval_level")}
            | {"probs": [m["probs"][f] for f in ("win", "gammon_win", "backgammon_win",
                                                  "gammon_loss", "backgammon_loss")]}
            for m in moves]


@pytest.mark.parametrize("level", ["2ply", "3ply"])
def test_single_and_batch_give_the_same_lists(level):
    a = BgBotAnalyzer(eval_level=level, parallel_threads=2)
    batch = _batch(a, CASES)
    for i, (b, d1, d2, kw) in enumerate(CASES):
        single = _as_batch_dicts(_single(a, b, d1, d2, kw))
        assert single == [dict(m, probs=list(m["probs"])) for m in batch[i]], (level, i)


def test_forced_boards_are_searched_at_two_ply():
    a = BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
    hits = 0
    for b, d1, d2, kw in CASES[:8]:
        moves = _single(a, b, d1, d2, kw)
        pruned = [m["board"] for m in moves if m["eval_level"] == "1-ply"]
        if not pruned:
            continue
        hits += 1
        forced = _single(a, b, d1, d2, kw, force_boards=[pruned[-1]])
        by_board = {tuple(m["board"]): m for m in forced}
        assert by_board[tuple(pruned[-1])]["eval_level"] == "2-ply"
        # The batch accepts the same option per row and agrees.
        batch = _batch(a, [(b, d1, d2, kw)], force=[[pruned[-1]]])[0]
        assert _as_batch_dicts(forced) == [dict(m, probs=list(m["probs"])) for m in batch]
    assert hits >= 3


def test_forced_boards_change_nothing_at_full_width():
    a = BgBotAnalyzer(eval_level="3ply", parallel_threads=2)
    b, d1, d2, kw = CASES[0]
    worst = _single(a, b, d1, d2, kw)[-1]["board"]
    assert _single(a, b, d1, d2, kw, force_boards=[worst]) == _single(a, b, d1, d2, kw)


def test_a_dead_cube_keeps_cubeless_values_for_pruned_moves():
    a = BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
    b, d1, d2 = CASES[0][:3]
    for m in _single(a, b, d1, d2, dict(cube_value=2, cube_owner="player", max_cube_value=2)):
        if m["eval_level"] == "1-ply":
            assert m["equity"] == m["cubeless_equity"]


@pytest.mark.parametrize("level", ["2ply", "3ply"])
def test_the_python_pass_no_longer_runs(level):
    a = BgBotAnalyzer(eval_level=level, parallel_threads=2)
    inner = a._analyzer._inner
    with patch.object(inner, "checker_play_analytics", wraps=inner.checker_play_analytics) as py, \
            patch.object(bgbot_cpp, "checker_play_nply", wraps=bgbot_cpp.checker_play_nply) as native:
        a.checker_play(*CASES[1][:3], **CASES[1][3])
    assert not py.called
    assert native.call_count == 1
