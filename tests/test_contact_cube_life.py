"""The contact cube life (``CubeInfo.contact_x``, ``contact_x_pip_slope``) and
the analyzer's per-depth money policy (``MONEY_CONTACT_CUBE_LIFE`` and
``MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE``)."""

from unittest.mock import patch

import bgbot_cpp
import pytest

import bgsage
from bgsage.analyzer import MONEY_CONTACT_CUBE_LIFE, MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE
from bgsage.weights import WeightConfig

# Money benchmark positions (player on roll, centred cube). At 1-ply the
# engine's default cube life doubles CONTACT; at 0.75 it does not.
CONTACT = [0, 0, 0, 0, 2, 2, 4, -2, 2, 0, 2, 0, -2, 3, -1, 0, -1, -2, 0, -2, -3, -2, 0, 0, 0, 0]
RACE = [0, 1, 1, 2, 2, 2, 3, 1, 0, 1, 2, 0, -1, 0, 0, 0, -2, -2, 0, -2, -1, -2, -1, -2, -2, 0]

KEYS = ("equity_nd", "equity_dt", "equity_dp", "should_double", "should_take")


@pytest.fixture(scope="module")
def weights():
    return WeightConfig.default()


def one_ply(weights, board, **kw):
    r = bgbot_cpp.evaluate_cube_decision_unified(
        board, 1, bgbot_cpp.CubeOwner.CENTERED, weights.strategy_type,
        weights.weight_paths_list, weights.hidden_sizes_list,
        jacoby=True, beaver=True, **kw)
    return {k: r[k] for k in KEYS}


def two_ply(weights, board, **kw):
    r = bgbot_cpp.cube_decision_nply_unified(
        board, 1, bgbot_cpp.CubeOwner.CENTERED, 2, weights.strategy_type,
        weights.weight_paths_list, weights.hidden_sizes_list,
        jacoby=True, beaver=True, **kw)
    return {k: r[k] for k in KEYS}


@pytest.mark.parametrize("evaluate", [one_ply, two_ply])
def test_default_contact_x_is_the_engine_default(weights, evaluate):
    # contact_x = 0.68 is cube_efficiency's contact value, so it must change
    # nothing, and -1 (the binding default) must equal leaving it out.
    base = evaluate(weights, CONTACT)
    assert evaluate(weights, CONTACT, contact_x=-1.0) == base
    assert evaluate(weights, CONTACT, contact_x=0.68) == base


@pytest.mark.parametrize("evaluate", [one_ply, two_ply])
def test_contact_x_leaves_races_alone(weights, evaluate):
    assert evaluate(weights, RACE, contact_x=0.95) == evaluate(weights, RACE)
    # cube_x_override, by contrast, does reach races.
    assert evaluate(weights, RACE, cube_x_override=0.95) != evaluate(weights, RACE)


def test_contact_x_changes_the_contact_decision(weights):
    assert one_ply(weights, CONTACT)["should_double"]
    assert not one_ply(weights, CONTACT, contact_x=0.75)["should_double"]


@pytest.mark.parametrize("level,n_plies,binding", [
    ("1ply", 1, "evaluate_cube_decision_unified"),
    ("2ply", 2, "cube_decision_nply_unified"),
    ("3ply", 3, "cube_decision_nply_unified"),
])
def test_analyzer_applies_the_money_policy_only(level, n_plies, binding):
    analyzer = bgsage.BgBotAnalyzer(eval_level=level, parallel_threads=1)
    native = getattr(bgbot_cpp, binding)
    with patch.object(bgbot_cpp, binding, side_effect=native) as evaluate:
        analyzer.cube_action(CONTACT)
        analyzer.cube_action(CONTACT, away1=5, away2=3)
    money, match = evaluate.call_args_list
    assert money.kwargs["contact_x"] == MONEY_CONTACT_CUBE_LIFE.get(n_plies, -1.0)
    assert money.kwargs["contact_x_pip_slope"] == (
        MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE.get(n_plies, 0.0)
        if n_plies in MONEY_CONTACT_CUBE_LIFE else 0.0)
    assert match.kwargs["contact_x"] == -1.0
    assert match.kwargs["contact_x_pip_slope"] == 0.0


def test_batch_one_ply_matches_the_analyzer():
    analyzer = bgsage.BgBotAnalyzer(eval_level="1ply")
    positions = [{"board": CONTACT, "cube_value": 1, "cube_owner": "centered"},
                 {"board": CONTACT, "cube_value": 1, "cube_owner": "centered",
                  "away1": 5, "away2": 3}]
    # One call per position: a batch holding any match position turns Jacoby
    # off for all of them.
    batch = [bgsage.batch_evaluate([p], eval_level="1ply", n_threads=1)[0] for p in positions]
    for pos, b in zip(positions, batch):
        a = analyzer.cube_action(CONTACT, away1=pos.get("away1", 0), away2=pos.get("away2", 0))
        assert b.equity_nd == pytest.approx(a.equity_nd, abs=1e-6)
        assert b.equity_dt == pytest.approx(a.equity_dt, abs=1e-6)
        assert b.optimal_action == a.optimal_action


@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_batch_checker_play_matches_the_analyzer(level):
    analyzer = bgsage.BgBotAnalyzer(eval_level=level, parallel_threads=1)
    a = analyzer.checker_play(CONTACT, 6, 2, cube_value=1, cube_owner="centered")
    b = bgsage.batch_checker_play(
        [{"board": CONTACT, "die1": 6, "die2": 2, "cube_value": 1, "cube_owner": "centered"}],
        eval_level=level, n_threads=1)[0]
    assert list(b.moves[0].board) == list(a.moves[0].board)
    assert b.moves[0].equity == pytest.approx(a.moves[0].equity, abs=1e-5)


def test_one_ply_cubeful_equity_follows_the_cube_life():
    # The depth's cube life reaches checker play too: at 1-ply the best move's
    # cubeful equity is exactly minus the opponent's 1-ply cube analysis.
    # (At 2-ply checker play routes the tree from the pre-move board and the
    # cube analysis does not, so the two differ on this position with or
    # without the cube life; test_checker_cube_symmetry covers N-ply.)
    analyzer = bgsage.BgBotAnalyzer(eval_level="1ply", parallel_threads=1)
    best = analyzer.checker_play(CONTACT, 6, 2, cube_value=1, cube_owner="centered").moves[0]
    opp = analyzer.cube_action(bgsage.flip_board(list(best.board)), cube_value=1,
                               cube_owner="centered")
    assert best.equity == pytest.approx(-opp.optimal_equity, abs=1e-6)


def test_batch_post_move_one_ply_matches_the_analyzer():
    analyzer = bgsage.BgBotAnalyzer(eval_level="1ply")
    post = bgsage.flip_board(CONTACT)
    a = analyzer.post_move_analytics(post, cube_owner="centered")
    b = bgsage.batch_post_move_evaluate([{"board": post, "cube_owner": "centered"}],
                                        eval_level="1ply", n_threads=1)[0]
    assert b.cubeful_equity == pytest.approx(a.cubeful_equity, abs=1e-5)


# ---------------------------------------------------------------------------
# contact_x_pip_slope
# ---------------------------------------------------------------------------


def test_contact_cube_x_formula():
    # contact_x at 120 average pips, + slope per pip of the average, clamped.
    assert bgbot_cpp.contact_cube_x(0.74, 0.002, 120, 120) == pytest.approx(0.74)
    assert bgbot_cpp.contact_cube_x(0.74, 0.002, 100, 160) == pytest.approx(0.76)
    # The average makes it the same from either side.
    assert (bgbot_cpp.contact_cube_x(0.74, 0.002, 70, 130)
            == bgbot_cpp.contact_cube_x(0.74, 0.002, 130, 70))
    assert bgbot_cpp.contact_cube_x(0.74, 0.01, 10, 10) == pytest.approx(0.5)
    assert bgbot_cpp.contact_cube_x(0.74, 0.01, 200, 200) == pytest.approx(1.0)
    assert bgbot_cpp.contact_cube_x(0.74, 0.0, 10, 10) == pytest.approx(0.74)


@pytest.mark.parametrize("evaluate", [one_ply, two_ply])
def test_zero_slope_is_flat_contact_x(weights, evaluate):
    assert (evaluate(weights, CONTACT, contact_x=0.75, contact_x_pip_slope=0.0)
            == evaluate(weights, CONTACT, contact_x=0.75))
    # Without contact_x the slope has nothing to act on.
    assert evaluate(weights, CONTACT, contact_x_pip_slope=0.003) == evaluate(weights, CONTACT)


@pytest.mark.parametrize("evaluate", [one_ply, two_ply])
def test_slope_leaves_races_alone(weights, evaluate):
    assert (evaluate(weights, RACE, contact_x=0.75, contact_x_pip_slope=0.003)
            == evaluate(weights, RACE))


def test_one_ply_slope_is_flat_x_at_the_positions_pips(weights):
    # A 1-ply cube decision is one Janowski conversion of the position itself,
    # so the slope must equal a flat contact_x at the value it resolves to.
    pp, op = bgbot_cpp.pip_counts(CONTACT)
    x = bgbot_cpp.contact_cube_x(0.70, 0.003, pp, op)
    assert x != pytest.approx(0.70)
    got = one_ply(weights, CONTACT, contact_x=0.70, contact_x_pip_slope=0.003)
    want = one_ply(weights, CONTACT, contact_x=x)
    for k in KEYS:
        assert got[k] == pytest.approx(want[k], abs=1e-6)


def test_two_ply_slope_reaches_the_leaves_and_the_cache_key(weights):
    # At 2-ply each leaf takes the cube life of its own pip count, and the
    # cubeful cache must not serve one slope's values to another.
    flat = two_ply(weights, CONTACT, contact_x=0.75)
    sloped = two_ply(weights, CONTACT, contact_x=0.75, contact_x_pip_slope=0.004)
    assert sloped["equity_nd"] != pytest.approx(flat["equity_nd"], abs=1e-6)
    assert two_ply(weights, CONTACT, contact_x=0.75) == flat
    assert two_ply(weights, CONTACT, contact_x=0.75, contact_x_pip_slope=0.004) == sloped


def test_one_ply_checker_equity_follows_the_sloped_cube_life():
    # With the slope on, the best move's 1-ply cubeful equity is still exactly
    # minus the opponent's 1-ply cube analysis (both take x from the board).
    assert MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE.get(1, 0.0) != 0.0
    analyzer = bgsage.BgBotAnalyzer(eval_level="1ply", parallel_threads=1)
    for d1, d2 in [(6, 2), (5, 5), (4, 1)]:
        best = analyzer.checker_play(CONTACT, d1, d2, cube_value=1,
                                     cube_owner="centered").moves[0]
        opp = analyzer.cube_action(bgsage.flip_board(list(best.board)), cube_value=1,
                                   cube_owner="centered")
        assert best.equity == pytest.approx(-opp.optimal_equity, abs=1e-6)
