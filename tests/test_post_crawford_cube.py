# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Post-Crawford trailer's automatic double in the cubeful match conversion.

After the Crawford game the trailer doubles at the first opportunity: the
leader is 1-away and cannot gain from the cube, and the game cannot end on the
trailer's side before the trailer's next pre-roll turn. A centered (or
trailer-owned) cube therefore behaves as if already doubled once.

The Janowski conversion (cl2cf_match) used to interpolate between a live cube
and a DEAD cube at the current value, where a trailer's gammon at 3-away/1-away
scores 2 points (to 1-away/1-away) instead of the match. With x ~ 0.68 that put
No Double 0.06-0.18 equity below Double/Take at 2-ply in 15 positions from
engine duels -- while both GNUbg (hint and 1296-trial rollout) and Neongammon's
own rollout give No Double == Double/Take there. The cube decision still said
"double" (the auto_double rule), but the No Double equity it reported was
wrong, so every analysis built on it charged the trailer an error for not
doubling.

Run with:
    PYTHONPATH="$PWD/build:$PWD/python" python -m pytest -q tests/test_post_crawford_cube.py
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, "build"))
sys.path.insert(0, os.path.join(project_dir, "python"))

import pytest  # noqa: E402

import bgbot_cpp as B  # noqa: E402
from neongammon import BgBotAnalyzer  # noqa: E402

C, P, O = B.CubeOwner.CENTERED, B.CubeOwner.PLAYER, B.CubeOwner.OPPONENT

# Pre-roll cubeless probabilities [win, win g, win bg, lose g, lose bg] of an
# ordinary contact position (the 2-ply probabilities of DUEL_3A_1A below).
CONTACT = [0.574, 0.136, 0.004, 0.101, 0.002]
# A trailer far too good: mostly gammons, so playing on at the current cube
# beats the cash the leader would take on a double.
GAMMONISH = [0.95, 0.85, 0.10, 0.01, 0.0]

# A 5-point engine duel (Neongammon vs GNUbg), score 2-4, post-Crawford; the
# trailer (3-away) is on roll with the cube centered on 1.
DUEL_3A_1A = [0, 0, 2, -1, -2, 2, 2, 2, 3, 0, 0, 0, 0, 2, -1, 0, -3, -2, 0,
              -2, -2, 2, 0, -2, 0, 0]
# Score 4-1 in the same kind of duel: trailer 4-away, leader 1-away.
DUEL_4A_1A = [0, 0, 0, 0, 0, 0, 5, -2, 3, 1, 0, 0, -3, 4, 0, 0, 0, -3, -2, -5,
              0, 1, 0, 0, 1, 0]

X_VALUES = (0.0, 0.68, 1.0)


def cubeful_mwc(probs, cube_value, owner, x, away1, away2, is_crawford=False):
    """cl2cf_match via the public binding, converted back to MWC."""
    eq = B.cl2cf(probs, cube_value, owner, x, away1, away2, is_crawford, False)
    return B.eq2mwc(eq, away1, away2, cube_value, is_crawford)


def cl(probs, away1, away2, cube_value):
    return B.cubeless_mwc(probs, away1, away2, cube_value, False)


# ---------------------------------------------------------------------------
# The conversion itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("probs", [CONTACT, GAMMONISH])
@pytest.mark.parametrize("x", X_VALUES)
def test_trailer_with_centered_cube_is_valued_as_auto_doubled(probs, x):
    # Trailer 3-away vs leader 1-away. Doubling now: the leader takes (cube 2,
    # dead) or passes (trailer to 2-away/1-away post-Crawford). The trailer
    # can also never double and play on at cube 1.
    take = cl(probs, 3, 1, 2)
    cash = B.get_met(2, 1, True)
    expected = max(cl(probs, 3, 1, 1), min(take, cash))
    assert cubeful_mwc(probs, 1, C, x, 3, 1) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("probs", [CONTACT, GAMMONISH])
@pytest.mark.parametrize("x", X_VALUES)
def test_leader_facing_a_trailer_with_cube_access_mirrors_it(probs, x):
    # Same match from the leader's side: the opponent doubles at their next
    # turn unless never doubling is better for them; the leader then takes
    # (cube 2) or passes (losing 1 point).
    take = cl(probs, 1, 3, 2)
    pass_ = B.get_met(1, 2, True)
    expected = min(cl(probs, 1, 3, 1), max(take, pass_))
    assert cubeful_mwc(probs, 1, C, x, 1, 3) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("x", X_VALUES)
def test_trailer_owned_cube_redoubles_automatically(x):
    # 4-away trailer owns a 2-cube: a redouble to 4 is automatic as well.
    take = cl(CONTACT, 4, 1, 4)
    cash = B.get_met(2, 1, True)
    expected = max(cl(CONTACT, 4, 1, 2), min(take, cash))
    assert cubeful_mwc(CONTACT, 2, P, x, 4, 1) == pytest.approx(expected, abs=1e-5)


def test_auto_double_matches_the_double_take_branch_when_the_leader_takes():
    # With an ordinary position the leader takes, so No Double carries the
    # value of the doubled cube -- exactly the Double/Take branch.
    nd = cubeful_mwc(CONTACT, 1, C, 0.68, 3, 1)
    dt = cubeful_mwc(CONTACT, 2, O, 0.68, 3, 1)
    assert nd == pytest.approx(dt, abs=1e-5)


# ---------------------------------------------------------------------------
# Unchanged neighbours
# ---------------------------------------------------------------------------

def test_crawford_game_stays_dead():
    for x in X_VALUES:
        assert cubeful_mwc(CONTACT, 1, C, x, 3, 1, is_crawford=True) == \
            pytest.approx(B.cubeless_mwc(CONTACT, 3, 1, 1, True), abs=1e-5)


def test_leader_owned_cube_after_the_double_stays_dead():
    for x in X_VALUES:
        assert cubeful_mwc(CONTACT, 2, O, x, 3, 1) == \
            pytest.approx(cl(CONTACT, 3, 1, 2), abs=1e-5)


def test_trailer_whose_win_already_takes_the_match_is_not_auto_doubled():
    # 2-away trailer with a 2-cube it owns: any win takes the match, so the
    # cube is dead and nothing is doubled.
    for x in X_VALUES:
        assert cubeful_mwc(CONTACT, 2, P, x, 2, 1) == \
            pytest.approx(cl(CONTACT, 2, 1, 2), abs=1e-5)


def test_pre_crawford_keeps_the_janowski_interpolation():
    values = [cubeful_mwc(CONTACT, 1, C, x, 3, 2) for x in X_VALUES]
    assert values[0] < values[1] < values[2]


# ---------------------------------------------------------------------------
# Cube decisions on real duel positions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eval_level", ["1ply", "2ply"])
@pytest.mark.parametrize("board,away1", [(DUEL_3A_1A, 3), (DUEL_4A_1A, 4)])
def test_no_double_equals_double_take_for_the_post_crawford_trailer(eval_level, board, away1):
    analyzer = BgBotAnalyzer(eval_level=eval_level)
    r = analyzer.cube_action(board, cube_value=1, cube_owner="centered",
                             away1=away1, away2=1, is_crawford=False,
                             jacoby=False, beaver=False)
    assert r.should_double
    assert r.should_take
    # Before the fix the gap was 0.097 (3-away) and 0.18 (4-away) at 2-ply.
    assert r.equity_nd == pytest.approx(min(r.equity_dt, r.equity_dp), abs=0.01)
