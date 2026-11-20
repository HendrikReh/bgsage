# SPDX-License-Identifier: MPL-2.0
"""N-ply batch_checker_play returns BgBotAnalyzer.checker_play's ranked list.

Both N-ply batch overloads used to run the cube-aware tree for every legal
move, so at 2-ply the pick could land on a move the analyzer's 1-ply filter
had rejected. Beyond that, they took the probs from the cubeless tree, routed
the cube tree per node instead of from the decision's root, and searched match
positions at cube value 1. They now go through the analyzer's steps one for
one (filter, cubeless search and promotion, cube tree for the searched moves,
1-ply Janowski for the rest, cubeful promotion), so the two lists are equal
field for field: with the bearoff database (both default to it) and without.
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

from bgsage import (  # noqa: E402
    BgBotAnalyzer, STARTING_BOARD, batch_checker_play, flip_board, possible_moves,
)
from bgsage.board import check_game_over  # noqa: E402

# A snake: the 24-NN layout pins every contact node of its tree to NN 22.
SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
# Both sides home: candidates and leaves come from the bearoff database.
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
# Two checkers still outside: some candidates are bearoff positions, some not.
NEAR_BEAROFF = [0, 2, 3, 0, 2, 3, 1, 0, 1, 1] + [0] * 9 + [-3, -2, -2, -3, -3, -2] + [0]

CUBES = [
    dict(cube_value=1, cube_owner="centered"),
    dict(cube_value=2, cube_owner="opponent"),
    dict(cube_value=2, cube_owner="player"),
    dict(cube_value=1, cube_owner="centered", away1=5, away2=3),
    dict(cube_value=2, cube_owner="player", away1=7, away2=4),
    dict(cube_value=1, cube_owner="centered", away1=1, away2=4, is_crawford=True),
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
                out.append(dict(board=board, die1=d1, die2=d2,
                                **CUBES[len(out) % len(CUBES)]))
    return out


FIXED = [dict(board=SNAKE, die1=6, die2=5, **CUBES[0]),
         dict(board=SNAKE, die1=2, die2=1, **CUBES[4]),
         dict(board=BEAROFF, die1=4, die2=2, **CUBES[3]),
         dict(board=BEAROFF, die1=6, die2=1, **CUBES[0]),
         dict(board=NEAR_BEAROFF, die1=5, die2=3, **CUBES[0]),
         dict(board=NEAR_BEAROFF, die1=6, die2=6, **CUBES[1])]
POSITIONS = {"1ply": _random_positions(36, seed=7) + FIXED,
             "2ply": _random_positions(36, seed=7) + FIXED,
             "3ply": _random_positions(12, seed=7) + FIXED}


def _analyzer_moves(level, positions, bearoff_db=True):
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=2,
                             bearoff_db=bearoff_db)
    out = []
    for p in positions:
        kw = {k: v for k, v in p.items() if k not in ("board", "die1", "die2")}
        out.append([asdict(m) for m in
                    analyzer.checker_play(p["board"], p["die1"], p["die2"], **kw).moves])
    return out


def _batch_moves(level, positions, n_threads=4, bearoff_db=None):
    return [[asdict(m) for m in r.moves]
            for r in batch_checker_play(positions, eval_level=level, n_threads=n_threads,
                                        bearoff_db=bearoff_db)]


@pytest.fixture(scope="module", params=["1ply", "2ply", "3ply"])
def lists(request):
    level = request.param
    positions = POSITIONS[level]
    return level, _batch_moves(level, positions), _analyzer_moves(level, positions)


def test_batch_returns_the_analyzers_list(lists):
    level, batch, analyzer = lists
    for i, (b, a) in enumerate(zip(batch, analyzer)):
        assert b == a, (level, i)


def test_two_ply_searches_only_the_filtered_moves(lists):
    # At 2-ply the moves the filter pruned stay on 1-ply values unless they
    # were promoted; at 3-ply every move is searched.
    level, batch, _ = lists
    labels = {m["eval_level"] for moves in batch for m in moves}
    assert labels == {"1ply": {"1-ply"}, "2ply": {"2-ply", "1-ply"}, "3ply": {"3-ply"}}[level]
    for moves in batch:
        assert all(m["eval_level"] == f"{level[0]}-ply" for m in moves[:2])


def test_the_fixture_exercises_promotion(lists):
    # The parity above covers the promotion loops only if moves outside the
    # 1-ply filter get searched: here several do, and some end up best.
    level, batch, _ = lists
    if level != "2ply":
        pytest.skip("only 2-ply filters and promotes")
    inner = BgBotAnalyzer(eval_level="2ply", cubeful=True, parallel_threads=2)._analyzer._inner
    promoted_best = promoted_any = 0
    for p, moves in zip(POSITIONS["2ply"], batch):
        kw = {k: p[k] for k in ("cube_value", "away1", "away2", "is_crawford") if k in p}
        scored = inner._score_candidates(
            possible_moves(p["board"], p["die1"], p["die2"]), p["board"],
            p["cube_owner"], **kw)
        _, survivors = inner._filter_candidates(
            scored, inner.FILTER_THRESHOLD, inner.FILTER_MAX_MOVES)
        promoted = [k for k, m in enumerate(moves)
                    if m["eval_level"] == "2-ply" and tuple(m["board"]) not in survivors]
        promoted_any += bool(promoted)
        promoted_best += 0 in promoted
    assert promoted_any >= 10
    assert promoted_best >= 1


def test_the_filtered_depths_come_from_python():
    # With no filtered depth, 2-ply searches every move, in the analyzer too.
    positions = POSITIONS["2ply"][:6]
    with patch("bgsage.batch.CUBEFUL_FILTERED_PLIES", frozenset()), \
            patch("bgsage.analyzer.CUBEFUL_FILTERED_PLIES", frozenset()):
        batch = _batch_moves("2ply", positions)
        analyzer = _analyzer_moves("2ply", positions)
    assert batch == analyzer
    assert all(m["eval_level"] == "2-ply" for moves in batch for m in moves)


def test_without_the_bearoff_database():
    positions = FIXED[2:]
    assert _batch_moves("2ply", positions, bearoff_db=False) == \
        _analyzer_moves("2ply", positions, bearoff_db=False)
    for db in (False, True):
        assert _batch_moves("1ply", positions, bearoff_db=db) == \
            _analyzer_moves("1ply", positions, bearoff_db=db)
    # And the database does change these positions.
    assert _batch_moves("2ply", positions, bearoff_db=False) != \
        _batch_moves("2ply", positions)


def test_thread_count_does_not_change_the_result():
    positions = POSITIONS["2ply"][:12]
    assert _batch_moves("2ply", positions, n_threads=1) == \
        _batch_moves("2ply", positions, n_threads=6)
