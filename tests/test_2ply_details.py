# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""
Tests for incl_2ply_details in cube action analytics.

The details are captured from the evaluation engine's own tree: the tree the
plain cube_action call walks, with the 21 player rolls (and, from 3-ply, the
21 opponent rolls below each) recorded as it goes. So:
- the headline equities equal the plain call's exactly;
- each roll's board is the engine's pick (1-ply cubeful against the cube
  state of the side to move in the ND scenario, PubEval keep-15 prefilter
  above 16 candidates, root routing), shared by the ND and DT sections;
- each ND per-roll equity is the opponent's (N-1)-ply cube analysis of that
  board, and each opponent-roll equity the player's (N-2)-ply one.

Run with:
    python -m pytest bgsage/tests/test_2ply_details.py -v
    python -m unittest bgsage.tests.test_2ply_details -v
"""
import os
import sys
import unittest

# Setup paths — tests/ lives at the bgsage repo root, so one level up is the repo.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)  # bgsage repo root
build_dir = os.path.join(repo_dir, "build")

if sys.platform == "win32":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(repo_dir, "python"))

import bgbot_cpp
from unittest import mock

import bgsage.analyzer
from bgsage import BgBotAnalyzer
from bgsage.board import flip_board, check_game_over, is_race, possible_moves

# Test position: contact/priming position with non-trivial cube dynamics
BOARD = [0, 0, 0, 2, 3, 0, 4, -2, 2, 0, 0, 0, -4, 2, -3, 0, -1, 0, 0, -3, 2, 0, -2, 0, 0, 0]

# Per-roll equity tolerances against standalone analyses of the same board.
#
# ND: the roll's subtree IS the standalone (N-1)-ply cube analysis of the
# opponent's position, so from 3-ply on the two agree exactly. At 2-ply the
# 1-ply leaf reuses the pick batch's grouped NN output (Accelerate, exact
# sigmoid) where the standalone 1-ply call evaluates the board on its own
# (lookup-table sigmoid): up to ~0.003 apart (observed 0.0024).
# Opponent rolls: the same one level deeper (a 1-ply leaf at 3-ply), doubled
# when the opponent's double is taken (observed 0.0063 there).
# DT: the tree makes one move pick per roll for both scenarios, against the
# ND cube state, while a standalone DT analysis (cube 2) picks against its
# own cube state, so their subtrees can choose different moves (observed
# 0.020 at 3-ply, 0.018 at 4-ply). A plumbing bug (wrong perspective,
# scaling, or roll mapping) produces diffs of 0.1+.
ND_TOL = {2: 0.005, 3: 1e-4, 4: 1e-4}
OPP_TOL = {3: 0.01, 4: 1e-4}
DT_TOL = {2: 0.006, 3: 0.03, 4: 0.03}

# The 21 dice combinations (matching C++ ALL_ROLLS order)
DICE_ROLLS = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6),
    (4, 5), (4, 6),
    (5, 6),
]


def assert_legal(test, before, d1, d2, after, msg):
    """`after` is a legal result of rolling d1-d2 from `before` (same
    perspective), standing pat included."""
    test.assertIn(list(after), [list(c) for c in possible_moves(before, d1, d2)], msg)


def setUpModule():
    # The per-roll checks compare an N-ply tree's interior against standalone
    # (N-1)- and (N-2)-ply analyses. The money cube life is depth-dependent
    # (MONEY_CONTACT_CUBE_LIFE), so a 2-ply tree's 1-ply leaf and a standalone
    # 1-ply analysis use different values by design. Pin every depth to the
    # engine default here so the checks compare like with like.
    _PATCHER.start()


def tearDownModule():
    _PATCHER.stop()


_PATCHER = mock.patch.dict(bgsage.analyzer.MONEY_CONTACT_CUBE_LIFE, clear=True)


class TestTwoPlyDetailsND(unittest.TestCase):
    """Verify ND section of 2-ply details against independent per-roll evaluations."""

    def _run_nd_test(self, n_plies: int):
        """Core ND test logic for a given ply level."""
        tol = ND_TOL[n_plies]
        analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)

        result = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )

        self.assertIsNotNone(result.details, "details should be present")
        self.assertIn("nd", result.details, "details should have 'nd' key")
        nd_rolls = result.details["nd"]
        self.assertEqual(len(nd_rolls), 21, "Should have 21 ND player rolls")

        # Verify weighted average of ND player-roll equities matches equity_nd
        weighted_sum = sum(
            pr["cubeful_equity"] * (1 if pr["die1"] == pr["die2"] else 2)
            for pr in nd_rolls
        )
        avg_equity = weighted_sum / 36.0
        self.assertAlmostEqual(
            avg_equity, result.equity_nd, places=3,
            msg=f"ND weighted average ({avg_equity:.6f}) should match equity_nd ({result.equity_nd:.6f})",
        )

        opp_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 1}ply", cubeful=True)
        # The opponent-roll level is captured only at 3-ply and above (at
        # 2-ply it would sit below the 1-ply leaf).
        player_cube_analyzer = (
            BgBotAnalyzer(eval_level=f"{n_plies - 2}ply", cubeful=True)
            if n_plies >= 3 else None
        )

        for roll_idx, pr in enumerate(nd_rolls):
            d1, d2 = DICE_ROLLS[roll_idx]
            self.assertEqual(pr["die1"], d1)
            self.assertEqual(pr["die2"], d2)

            # --- The player's board: a legal play of this roll ---
            assert_legal(self, BOARD, d1, d2, pr["checkers"],
                         f"Roll {d1}-{d2}: post-move board is not a legal play")
            if check_game_over(list(pr["checkers"])) != 0:
                self.assertNotIn(
                    "opponent_rolls", pr,
                    f"Roll {d1}-{d2}: terminal should not have opponent_rolls",
                )
                continue

            # --- Verify player-roll equity via opponent's cube action ---
            post_move_board = list(pr["checkers"])
            opp_board = flip_board(post_move_board)

            opp_cube = opp_cube_analyzer.cube_action(
                opp_board, cube_value=1, cube_owner="centered",
                jacoby=True, beaver=True,
            )

            expected_player_eq = -opp_cube.optimal_equity
            self.assertAlmostEqual(
                pr["cubeful_equity"], expected_player_eq, delta=tol,
                msg=f"Roll {d1}-{d2}: ND player equity {pr['cubeful_equity']:.4f} "
                    f"vs expected {expected_player_eq:.4f} "
                    f"(opp optimal_action={opp_cube.optimal_action})",
            )

            if n_plies == 2:
                # No opponent-roll level at 2-ply: the field must be absent.
                self.assertNotIn(
                    "opponent_rolls", pr,
                    f"Roll {d1}-{d2}: 2-ply details should not carry opponent_rolls",
                )
                continue

            is_opp_dp = (opp_cube.should_double and not opp_cube.should_take)

            if is_opp_dp:
                self.assertTrue(
                    pr.get("opponent_dp", False) or "opponent_rolls" not in pr,
                    f"Roll {d1}-{d2}: opponent D/P but opponent_rolls present",
                )
            else:
                self.assertIn(
                    "opponent_rolls", pr,
                    f"Roll {d1}-{d2}: no D/P but opponent_rolls missing",
                )
                opp_rolls = pr["opponent_rolls"]
                self.assertEqual(len(opp_rolls), 21)

                for opp_idx, opp_r in enumerate(opp_rolls):
                    od1, od2 = DICE_ROLLS[opp_idx]
                    self.assertEqual(opp_r["die1"], od1)
                    self.assertEqual(opp_r["die2"], od2)

                    assert_legal(self, opp_board, od1, od2, flip_board(opp_r["checkers"]),
                                 f"Roll {d1}-{d2}, opp {od1}-{od2}: not a legal play")

                    player_post_opp_board = list(opp_r["checkers"])
                    opp_post = flip_board(player_post_opp_board)
                    game_over_opp = check_game_over(opp_post)
                    if game_over_opp != 0:
                        continue

                    player_cube = player_cube_analyzer.cube_action(
                        player_post_opp_board, cube_value=1, cube_owner="centered",
                        jacoby=True, beaver=True,
                    )

                    if opp_cube.should_double and opp_cube.should_take:
                        player_cube_dt = player_cube_analyzer.cube_action(
                            player_post_opp_board, cube_value=2, cube_owner="player",
                            jacoby=True, beaver=True,
                        )
                        expected_opp_eq = 2.0 * player_cube_dt.optimal_equity
                    else:
                        expected_opp_eq = player_cube.optimal_equity

                    self.assertAlmostEqual(
                        opp_r["cubeful_equity"], expected_opp_eq, delta=OPP_TOL[n_plies],
                        msg=f"Roll {d1}-{d2}, opp {od1}-{od2}: "
                            f"equity {opp_r['cubeful_equity']:.4f} "
                            f"vs expected {expected_opp_eq:.4f}",
                    )

    def test_2ply_nd(self):
        """Test ND 2-ply details at 2-ply (true 2-ply, no opponent rolls)."""
        self._run_nd_test(2)

    def test_3ply_nd(self):
        """Test ND 2-ply details at 3-ply."""
        self._run_nd_test(3)

    def test_4ply_nd(self):
        """Test ND 2-ply details at 4-ply."""
        self._run_nd_test(4)


class TestTwoPlyDetailsDT(unittest.TestCase):
    """Verify DT section of 2-ply details against independent per-roll evaluations."""

    def _run_dt_test(self, n_plies: int):
        """Core DT test logic for a given ply level."""
        tol = DT_TOL[n_plies]
        analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)

        result = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )

        self.assertIsNotNone(result.details, "details should be present")
        self.assertIn("dt", result.details, "details should have 'dt' key")
        dt_rolls = result.details["dt"]
        self.assertEqual(len(dt_rolls), 21, "Should have 21 DT player rolls")

        # Verify weighted average of DT player-roll equities matches equity_dt
        weighted_sum = sum(
            pr["cubeful_equity"] * (1 if pr["die1"] == pr["die2"] else 2)
            for pr in dt_rolls
        )
        avg_equity = weighted_sum / 36.0
        self.assertAlmostEqual(
            avg_equity, result.equity_dt, places=3,
            msg=f"DT weighted average ({avg_equity:.6f}) should match equity_dt ({result.equity_dt:.6f})",
        )

        # Boards match ND: one move pick per roll serves both scenarios
        nd_rolls = result.details["nd"]
        for roll_idx in range(21):
            self.assertEqual(
                dt_rolls[roll_idx]["checkers"], nd_rolls[roll_idx]["checkers"],
                f"Roll {roll_idx}: DT and ND boards should be identical",
            )

        # For DT, opponent's cube action is at (n-1)-ply with cube=2, opponent owns
        opp_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 1}ply", cubeful=True)

        for roll_idx, pr in enumerate(dt_rolls):
            d1, d2 = DICE_ROLLS[roll_idx]
            if check_game_over(list(pr["checkers"])) != 0:
                continue

            post_move_board = list(pr["checkers"])
            opp_board = flip_board(post_move_board)

            # In DT scenario, cube is 2, opponent owns
            opp_cube = opp_cube_analyzer.cube_action(
                opp_board, cube_value=2, cube_owner="player",
                jacoby=True, beaver=True,
            )

            # Player's equity per initial cube = 2 * (-opp_optimal_equity)
            expected_player_eq = 2.0 * (-opp_cube.optimal_equity)
            self.assertAlmostEqual(
                pr["cubeful_equity"], expected_player_eq, delta=tol,
                msg=f"Roll {d1}-{d2}: DT player equity {pr['cubeful_equity']:.4f} "
                    f"vs expected {expected_player_eq:.4f}",
            )

    def test_2ply_dt(self):
        """Test DT 2-ply details at 2-ply (true 2-ply, no opponent rolls)."""
        self._run_dt_test(2)

    def test_3ply_dt(self):
        """Test DT 2-ply details at 3-ply."""
        self._run_dt_test(3)

    def test_4ply_dt(self):
        """Test DT 2-ply details at 4-ply."""
        self._run_dt_test(4)


class TestDetailsHeadlineMatchesPlain(unittest.TestCase):
    """The headline equities of a details call equal the plain call's.

    The details are captured from the very tree the plain call walks, so the
    two agree exactly. (Regression: before 2026-06 a 2-ply details request ran
    3-ply-grade levels, ~0.06 off plain 2-ply; until 2026-11 the details ran
    on an older recursion with cubeless picks and drifted up to 0.05.)
    """

    def _run_headline_test(self, n_plies: int):
        analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)
        plain = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True,
        )
        detailed = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )
        for field in ("equity_nd", "equity_dt", "equity_dp", "should_double",
                      "should_take", "optimal_action"):
            self.assertEqual(
                getattr(detailed, field), getattr(plain, field),
                msg=f"{n_plies}-ply {field}: details {getattr(detailed, field)!r} "
                    f"vs plain {getattr(plain, field)!r}",
            )

    def test_2ply_headline(self):
        """2-ply details headline == plain 2-ply (the original bug)."""
        self._run_headline_test(2)

    def test_3ply_headline(self):
        """3-ply details headline == plain 3-ply."""
        self._run_headline_test(3)

    def test_4ply_headline(self):
        """4-ply details headline == plain 4-ply."""
        self._run_headline_test(4)

    def test_2ply_headline_dancing(self):
        """2-ply headline equality on a position with dancing rolls.

        Player on the bar against a 5-point board (only a 1 enters), so the
        no-legal-moves branch of the 2-ply details path is exercised.
        """
        dancing_board = [
            0,
            0, 0, 0, 0, -2, 5, 0, 4, 0, 0, 0, 0,
            3, 0, 0, 0, -3, 0, -2, -2, -2, -2, -2, 2,
            1,
        ]
        analyzer = BgBotAnalyzer(eval_level="2ply", cubeful=True)
        plain = analyzer.cube_action(
            dancing_board, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True,
        )
        detailed = analyzer.cube_action(
            dancing_board, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )
        for field in ("equity_nd", "equity_dt"):
            self.assertEqual(
                getattr(detailed, field), getattr(plain, field),
                msg=f"dancing {field}: details {getattr(detailed, field)!r} "
                    f"vs plain {getattr(plain, field)!r}",
            )
        # A roll without a 1 dances: standing pat, board unchanged.
        for pr in detailed.details["nd"]:
            if pr["die1"] == 2 and pr["die2"] == 3:
                self.assertEqual(list(pr["checkers"]), dancing_board)
                self.assertNotIn("opponent_rolls", pr)
                break
        else:
            self.fail("Roll 2-3 not found in nd details")


class TestDetailsAcrossCubeStates(unittest.TestCase):
    """Headline and per-roll averages across cube states and root routing."""

    # A snake: the 24-NN layout pins every contact node of its tree to NN 22.
    SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
    CASES = [
        (BOARD, dict(cube_value=1, cube_owner="centered", away1=5, away2=3)),
        (BOARD, dict(cube_value=2, cube_owner="player", away1=7, away2=4)),
        (BOARD, dict(cube_value=1, cube_owner="centered", away1=1, away2=4, is_crawford=True)),
        (BOARD, dict(cube_value=2, cube_owner="opponent")),
        (SNAKE, dict(cube_value=1, cube_owner="centered")),
    ]

    def test_headline_and_averages(self):
        for n_plies in (2, 3):
            analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)
            for board, kw in self.CASES:
                plain = analyzer.cube_action(board, **kw)
                detailed = analyzer.cube_action(board, incl_2ply_details=True, **kw)
                label = f"{n_plies}-ply {kw}"
                for field in ("equity_nd", "equity_dt", "equity_dp", "optimal_action"):
                    self.assertEqual(getattr(detailed, field), getattr(plain, field),
                                     f"{label}: {field}")
                for section, headline in (("nd", detailed.equity_nd), ("dt", None)):
                    rolls = detailed.details[section]
                    self.assertEqual(len(rolls), 21, label)
                    if headline is None:
                        continue   # equity_dt may carry the beaver
                    avg = sum(r["cubeful_equity"] * (1 if r["die1"] == r["die2"] else 2)
                              for r in rolls) / 36.0
                    self.assertAlmostEqual(avg, headline, places=4, msg=label)
                for nd, dt in zip(detailed.details["nd"], detailed.details["dt"]):
                    self.assertEqual(nd["checkers"], dt["checkers"], label)
                    assert_legal(self, board, nd["die1"], nd["die2"], nd["checkers"], label)


class TestTwoPlyDetailsErrors(unittest.TestCase):
    """Test error handling for 2-ply details."""

    def test_error_below_2ply(self):
        """Verify that incl_2ply_details raises an error for 1-ply.

        2-ply and above are allowed: at 2-ply the details path captures only
        the player-roll level, evaluating each post-move position at the
        1-ply Janowski leaf (true 2-ply — headline matches the plain call).
        """
        analyzer = BgBotAnalyzer(eval_level="1ply", cubeful=True)
        with self.assertRaises(Exception):
            analyzer.cube_action(
                BOARD, cube_value=1, cube_owner="centered",
                incl_2ply_details=True,
            )

    def test_2ply_details_allowed(self):
        """Verify that incl_2ply_details works at 2-ply.

        At 2-ply only the player-roll level is captured (the opponent-roll
        level would sit below the 1-ply leaf), so opponent_rolls is absent
        from every entry.
        """
        analyzer = BgBotAnalyzer(eval_level="2ply", cubeful=True)
        result = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )
        self.assertIsNotNone(result.details)
        self.assertEqual(len(result.details["nd"]), 21)
        self.assertEqual(len(result.details["dt"]), 21)
        for section in ("nd", "dt"):
            for pr in result.details[section]:
                self.assertNotIn("opponent_rolls", pr)


if __name__ == "__main__":
    unittest.main()
