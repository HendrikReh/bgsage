# SPDX-License-Identifier: MPL-2.0
"""Batch functions apply the money-game rules (Jacoby, beaver) per position.

They used to turn Jacoby (and, in ``batch_evaluate``, the beaver) off for the
whole batch as soon as one row was a match, so a money position's result
depended on what else was in the batch. The native batch functions already
apply both rules only to money positions (``CubeInfo::jacoby_active`` checks
``is_money()``; the beaver lives on the money paths), so a mixed batch must
give every row exactly what it gives that row alone. Needs no external
benchmark files.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bgsage import (  # noqa: E402
    STARTING_BOARD, batch_checker_play, batch_evaluate, batch_post_move_evaluate,
    flip_board,
)
from bgsage.batch import batch_cube_action  # noqa: E402

CONTACT = [0, 0, 0, 0, 2, 2, 4, -2, 2, 0, 2, 0, -2, 3, -1, 0, -1, -2, 0, -2, -3, -2, 0, 0, 0, 0]

# The starting position's Double/Take equity is negative for the doubler, so
# with beavers on it is a Double/Beaver: the row where the beaver flag matters.
MONEY = [
    {"board": STARTING_BOARD, "cube_value": 1, "cube_owner": "centered"},
    {"board": CONTACT, "cube_value": 1, "cube_owner": "centered"},
    {"board": flip_board(CONTACT), "cube_value": 2, "cube_owner": "player"},
]
MATCH = [
    {"board": STARTING_BOARD, "cube_value": 1, "cube_owner": "centered",
     "away1": 5, "away2": 3},
    {"board": CONTACT, "cube_value": 1, "cube_owner": "centered",
     "away1": 1, "away2": 4, "is_crawford": True},
]
DICE = [(3, 1), (6, 2), (5, 5), (4, 2), (6, 6)]


def _with_dice(positions):
    return [dict(p, die1=d1, die2=d2) for p, (d1, d2) in zip(positions, DICE)]


def _pre_roll(r):
    return (r.probs, r.cubeless_equity, r.cubeful_equity, r.equity_nd, r.equity_dt,
            r.equity_dp, r.should_double, r.should_take, r.optimal_action,
            r.is_beaver)


def _post_move(r):
    return (r.cubeless_equity, r.cubeful_equity, r.eval_level)


def _checker(r):
    return [(tuple(m.board), m.equity, m.cubeless_equity, m.equity_diff,
             m.eval_level) for m in r.moves]


APIS = {
    "batch_evaluate": (batch_evaluate, _pre_roll, False),
    "batch_post_move_evaluate": (batch_post_move_evaluate, _post_move, False),
    "batch_checker_play": (batch_checker_play, _checker, True),
}


def _run(api, positions, level, **kw):
    fn, key, _ = APIS[api]
    return [key(r) for r in fn(positions, eval_level=level, n_threads=1, **kw)]


def _inputs(api, positions):
    return _with_dice(positions) if APIS[api][2] else positions


@pytest.mark.parametrize("level", ["1ply", "2ply"])
@pytest.mark.parametrize("api", list(APIS))
def test_mixed_batch_rows_match_the_row_alone(api, level):
    rows = _inputs(api, MONEY + MATCH)
    alone = [_run(api, [p], level)[0] for p in rows]
    assert _run(api, rows, level) == alone
    # The order of the batch does not matter either.
    assert _run(api, rows[::-1], level) == alone[::-1]


@pytest.mark.parametrize("level", ["1ply", "2ply"])
@pytest.mark.parametrize("api", list(APIS))
def test_match_rows_ignore_the_money_rules(api, level):
    """A match row gives the same result whatever jacoby/beaver say, so
    passing the flags through cannot change a match-only batch."""
    rows = _inputs(api, MATCH)
    kw_off = {"jacoby": False}
    if api == "batch_evaluate":
        kw_off["beaver"] = False
    assert _run(api, rows, level) == _run(api, rows, level, **kw_off)


def test_the_beaver_survives_a_mixed_batch():
    money, match = MONEY[0], MATCH[0]
    alone = batch_evaluate([money], n_threads=1)[0]
    assert alone.is_beaver, "fixture: the starting position should be a beaver"
    mixed = batch_evaluate([money, match], n_threads=1)[0]
    assert mixed.is_beaver
    assert mixed.equity_dt == alone.equity_dt
    cube = batch_cube_action([money, match], n_threads=1)[0]
    assert cube.is_beaver and cube.equity_dt == alone.equity_dt
