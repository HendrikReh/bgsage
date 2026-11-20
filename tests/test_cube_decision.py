# SPDX-License-Identifier: MPL-2.0
"""BgBotAnalyzer.cube_decision equals cube_action without the probabilities.

At 2-ply and deeper cube_action runs a second, cubeless N-ply search for its
probs and cubeless equity after the cube tree has decided. cube_decision
skips that search (``decision_only`` in cube_decision_nply_unified); its
equities and flags must equal cube_action's, and the simulation, which reads
only the flags, uses it and records the same games.
"""

import os
import random
import sys
from dataclasses import asdict

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bgsage import BgBotAnalyzer, STARTING_BOARD, flip_board, possible_moves  # noqa: E402
from bgsage.board import check_game_over  # noqa: E402
from bgsage.simulation import GameConfig, simulate_game  # noqa: E402

SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
CONTEXTS = [
    dict(),
    dict(jacoby=False, beaver=False),
    dict(cube_value=2, cube_owner="player"),
    dict(cube_value=2, cube_owner="player", max_cube_value=4),
    dict(away1=5, away2=3),
    dict(away1=4, away2=1),
    dict(away1=1, away2=4, is_crawford=True),
]
FIELDS = ("equity_nd", "equity_dt", "equity_dp", "should_double", "should_take",
          "optimal_equity", "optimal_action", "eval_level", "is_beaver")


def _positions(n, seed):
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
            out.append(b)
    return out


BOARDS = _positions(5, seed=772) + [SNAKE, BEAROFF]


@pytest.mark.parametrize("level", ["1ply", "2ply", "3ply"])
def test_cube_decision_equals_cube_action(level):
    a = BgBotAnalyzer(eval_level=level, parallel_threads=2)
    for i, board in enumerate(BOARDS):
        for kw in CONTEXTS:
            full = asdict(a.cube_action(board, **kw))
            lean = asdict(a.cube_decision(board, **kw))
            assert {k: lean[k] for k in FIELDS} == {k: full[k] for k in FIELDS}, (level, i, kw)
            assert set(lean) == set(FIELDS)


def test_cube_decision_equals_cube_action_at_four_ply():
    a = BgBotAnalyzer(eval_level="4ply", parallel_threads=4)
    for board, kw in ((BOARDS[0], CONTEXTS[0]), (SNAKE, CONTEXTS[0]), (BOARDS[1], CONTEXTS[4])):
        full, lean = asdict(a.cube_action(board, **kw)), asdict(a.cube_decision(board, **kw))
        assert {k: lean[k] for k in FIELDS} == {k: full[k] for k in FIELDS}


class _CubeActionOnly:
    """A policy without cube_decision: the simulation falls back to cube_action."""

    def __init__(self, analyzer):
        self._a = analyzer

    def cube_action(self, *args, **kw):
        return self._a.cube_action(*args, **kw)

    def checker_play(self, *args, **kw):
        return self._a.checker_play(*args, **kw)


@pytest.mark.parametrize("config", [GameConfig(), GameConfig(match_length=5, score1=2, score2=3)])
def test_seeded_simulation_records_do_not_change(config):
    a = BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
    for seed in (3, 11):
        lean = simulate_game(seed=seed, config=config, players=(a, a))
        full = simulate_game(seed=seed, config=config,
                             players=(_CubeActionOnly(a), _CubeActionOnly(a)))
        assert lean == full, seed
