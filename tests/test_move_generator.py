# SPDX-License-Identifier: MPL-2.0
"""The move generators: dice rules, and their exact output pinned.

possible_moves (sorted) and possible_moves_unsorted (first-seen order, what
the evaluation hot paths see) share one implementation of the dice rules.
The digest pins both outputs, including the unsorted order that breaks exact
ties downstream, over every roll of fixed boards: it was computed by the
generators before the rules were shared and must not change unless the
generator is meant to.
"""

import hashlib
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bgbot_cpp  # noqa: E402

from neongammon import STARTING_BOARD, flip_board  # noqa: E402

ORDERED_ROLLS = [(a, b) for a in range(1, 7) for b in range(1, 7)]


def _board(points, bar=0, opp=None):
    b = [0] * 26
    for p, n in points.items():
        b[p] = n
    for p, n in (opp or {}).items():
        b[p] = -n
    b[25] = bar
    return b


CLOSED = _board({6: 5, 8: 3, 13: 6}, bar=1,
                opp={19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 12: 3})
LAST_CHECKER = _board({2: 1}, opp={24: 5, 23: 5, 22: 5})


def _random_boards(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = list(STARTING_BOARD)
        for _ in range(rng.randint(2, 70)):
            moves = bgbot_cpp.possible_moves(b, rng.randint(1, 6), rng.randint(1, 6))
            if moves:
                b = list(rng.choice(moves))
            if not any(x > 0 for x in b[1:26]):
                break
            b = list(flip_board(b))
        else:
            out.append(b)
    return out


BOARDS = [list(STARTING_BOARD), CLOSED, LAST_CHECKER] + _random_boards(100, seed=3)


def test_both_generators_give_the_same_boards():
    for b in BOARDS:
        for d1, d2 in ORDERED_ROLLS:
            s = bgbot_cpp.possible_moves(b, d1, d2)
            u = bgbot_cpp.possible_moves_unsorted(b, d1, d2)
            assert s == sorted(s)
            assert len(set(map(tuple, u))) == len(u)
            assert sorted(map(tuple, u)) == list(map(tuple, s))


def test_dice_rules():
    # Standing pat: no legal move leaves the board unchanged.
    for d1, d2 in ORDERED_ROLLS:
        assert bgbot_cpp.possible_moves(CLOSED, d1, d2) == [CLOSED]
    # The last checker comes off with the first die; the game is over.
    assert bgbot_cpp.possible_moves(LAST_CHECKER, 6, 5) == [LAST_CHECKER[:2] + [0] + LAST_CHECKER[3:]]
    # Opening 3-1 makes the 5-point among its 16 plays.
    moves = bgbot_cpp.possible_moves(STARTING_BOARD, 3, 1)
    assert len(moves) == 16
    assert [0, -2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0] in moves


def _digest():
    h = hashlib.sha256()
    for b in BOARDS:
        for d1, d2 in ORDERED_ROLLS:
            h.update(json.dumps([bgbot_cpp.possible_moves(b, d1, d2),
                                 bgbot_cpp.possible_moves_unsorted(b, d1, d2)]).encode())
    return h.hexdigest()


# Computed with the generators before the dice rules were shared (main at
# 0e179f9 plus the possible_moves_unsorted binding), and identical after.
PINNED_DIGEST = "d3a65fb6d65d7957a6dc8051acd6b18726afd7f8267c6fe0800bfbc342410926"


def test_the_output_is_pinned():
    assert _digest() == PINNED_DIGEST
