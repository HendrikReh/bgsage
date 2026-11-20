"""Verify that cubeless probability clamping enforces position invariants.

The multi-ply and truncated rollout evaluators must zero out impossible
outcomes:
  1. P(gl) and P(bl) when the player has at least one checker borne off.
  2. P(gw) and P(bw) when the opponent has at least one checker borne off.
  3. P(bl) when contact is broken AND the player has no checker in the
     opponent's home (19-24) or on bar (25).
  4. P(bw) when contact is broken AND the opponent has no checker in the
     player's home (1-6) or on bar (0).

This script picks positions that exercise each invariant and checks that
1-ply, 2-ply, 3-ply, and truncated/full rollout evaluations return probs
that match.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "build"))
sys.path.insert(0, str(ROOT / "python"))


import bgbot_cpp
from bgsage import BgBotAnalyzer, STARTING_BOARD


def fmt_probs(probs):
    return (
        f"win={probs.win:.4f} gw={probs.gammon_win:.4f} bw={probs.backgammon_win:.4f} "
        f"gl={probs.gammon_loss:.4f} bl={probs.backgammon_loss:.4f}"
    )


def position_player_borne_off():
    """Player has 1 checker borne off, 14 still on the board.
    Opponent still has all 15 checkers. Contact has been broken.
    Expected: P(gl) = P(bl) = 0 (player can't be gammoned/backgammoned).
    """
    # 14 player checkers in their home (so 1 borne off), opp still mid-bear-in.
    board = [0] * 26
    board[1] = 5
    board[2] = 4
    board[3] = 3
    board[4] = 2
    # Player total: 5+4+3+2 = 14, so 1 borne off.
    board[20] = -3
    board[21] = -3
    board[22] = -3
    board[23] = -3
    board[24] = -3
    # Opponent total: 15.
    return board


def position_opp_borne_off():
    """Opponent has 1 checker borne off; player has all 15.
    Contact has been broken.
    Expected: P(gw) = P(bw) = 0.
    """
    board = [0] * 26
    board[2] = 5
    board[3] = 5
    board[4] = 5
    # Player total: 15.
    board[20] = -4
    board[21] = -3
    board[22] = -3
    board[23] = -2
    board[24] = -2
    # Opponent total: 14, so 1 borne off.
    return board


def position_no_bg_loss():
    """Pure race, player has no checker in opp's home (19-24) or bar.
    Player has 15 checkers, all in their own home/outer.
    Opponent still has checkers in their home -- so opp could still gammon
    or backgammon... but player has nothing in opp's home, so backgammon
    LOSS is impossible (P(bl)=0). Gammon loss IS still possible.
    """
    board = [0] * 26
    board[6] = 5
    board[7] = 5
    board[8] = 5
    # Player total: 15, all on their home/outer (none on points 19-24 or bar).
    board[19] = -5
    board[20] = -5
    board[21] = -5
    # Opponent total: 15.
    return board


def position_no_bg_win():
    """Pure race, opponent has no checker in player's home (1-6) or bar.
    Symmetric to position_no_bg_loss.
    Expected: P(bw) = 0 (opponent can't be backgammoned by player).
    """
    board = [0] * 26
    board[7] = 5
    board[8] = 5
    board[9] = 5
    board[18] = -5
    board[19] = -5
    board[20] = -5
    return board


def check_invariant(label, board, *, expect_gl_zero=False, expect_bl_zero=False,
                    expect_gw_zero=False, expect_bw_zero=False):
    print(f"\n=== {label} ===")
    print(f"Board: {board}")

    levels = [
        ("1-ply", dict(eval_level="1ply", cubeful=False)),
        ("2-ply", dict(eval_level="2ply", cubeful=False)),
        ("3-ply", dict(eval_level="3ply", cubeful=False)),
        ("1T",    dict(eval_level="rollout", n_trials=42,  truncation_depth=5,
                        decision_ply=1, late_ply=-1, late_threshold=20,
                        ultra_late_threshold=2, cubeful=False)),
        ("2T",    dict(eval_level="rollout", n_trials=360, truncation_depth=7,
                        decision_ply=2, late_ply=1, late_threshold=2,
                        ultra_late_threshold=2, cubeful=False)),
        ("R",     dict(eval_level="rollout", n_trials=36, truncation_depth=0,
                        decision_ply=1, ultra_late_threshold=9999, cubeful=False)),
    ]

    all_ok = True
    for level_name, kwargs in levels:
        analyzer = BgBotAnalyzer(**kwargs)
        result = analyzer.post_move_analytics(board, cube_owner="centered")
        probs = result.probs
        problems = []
        if expect_gl_zero and probs.gammon_loss != 0.0:
            problems.append(f"P(gl)={probs.gammon_loss:.6f} (should be 0)")
        if expect_bl_zero and probs.backgammon_loss != 0.0:
            problems.append(f"P(bl)={probs.backgammon_loss:.6f} (should be 0)")
        if expect_gw_zero and probs.gammon_win != 0.0:
            problems.append(f"P(gw)={probs.gammon_win:.6f} (should be 0)")
        if expect_bw_zero and probs.backgammon_win != 0.0:
            problems.append(f"P(bw)={probs.backgammon_win:.6f} (should be 0)")
        status = "OK " if not problems else "FAIL"
        print(f"  [{status}] {level_name:5s}: {fmt_probs(probs)}")
        if problems:
            for p in problems:
                print(f"        - {p}")
            all_ok = False
    return all_ok


def main():
    overall = True
    overall &= check_invariant(
        "Player has 1 borne off (gl=bl=0)",
        position_player_borne_off(),
        expect_gl_zero=True, expect_bl_zero=True,
    )
    overall &= check_invariant(
        "Opponent has 1 borne off (gw=bw=0)",
        position_opp_borne_off(),
        expect_gw_zero=True, expect_bw_zero=True,
    )
    overall &= check_invariant(
        "Pure race, player not in opp's home (bl=0)",
        position_no_bg_loss(),
        expect_bl_zero=True,
    )
    overall &= check_invariant(
        "Pure race, opp not in player's home (bw=0)",
        position_no_bg_win(),
        expect_bw_zero=True,
    )
    print()
    if overall:
        print("All clamping invariants pass.")
        return 0
    else:
        print("FAIL: at least one position violates the invariants.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
