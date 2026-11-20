"""Full checker play analytics for 5-2 on the 8/6 8/3 reference position
   at 3T (XGRoller++ Checker), top 5 moves, money + match (4-away vs 1-away).

Mirrors checker_play_3P_full.py but with eval_level="truncated3".
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "build"))
sys.path.insert(0, str(REPO / "python"))

from bgsage import BgBotAnalyzer
from bgsage.text_export import compute_move_notation

BOARD = [0, 2, 2, 1, 2, 0, 2, 0, 2, 1, 0, 0,
         -4, 2, 0, 0, 0, -3, 0, -2, -4, 1, -2, 0, 0, 0]
D1, D2 = 5, 2


def report(analyzer, label, **kwargs):
    print("=" * 100)
    print(f"  {label}")
    print("=" * 100)
    t0 = time.time()
    r = analyzer.checker_play(BOARD, D1, D2,
                              cube_value=2, cube_owner="opponent", **kwargs)
    dt = time.time() - t0
    print(f"  wall: {dt:.2f}s   ({len(r.moves)} legal moves total)")
    print()

    # Header
    print(f"  {'#':<2} {'Move':<22} {'cfEq':>9} {'clEq':>9} {'diff':>9}  "
          f"{'Win':>7} {'gW':>7} {'bW':>7} {'gL':>7} {'bL':>7}  ply")
    print("  " + "-" * 96)

    best_cf = r.moves[0].equity
    for i, m in enumerate(r.moves[:5]):
        notation = compute_move_notation(BOARD, list(m.board), D1, D2)
        diff = "(best)" if i == 0 else f"{m.equity - best_cf:+.4f}"
        print(
            f"  {i+1:<2} {notation:<22} "
            f"{m.equity:>+9.5f} {m.cubeless_equity:>+9.5f} {diff:>9}  "
            f"{m.probs.win:>7.4f} {m.probs.gammon_win:>7.4f} "
            f"{m.probs.backgammon_win:>7.4f} "
            f"{m.probs.gammon_loss:>7.4f} {m.probs.backgammon_loss:>7.4f}  "
            f"{m.eval_level}"
        )
    print()


def main(n_threads: int = 16) -> None:
    analyzer = BgBotAnalyzer(eval_level="truncated3", cubeful=True,
                             parallel_threads=n_threads)
    print(f"\nReference position (player POV): {BOARD}")
    print(f"Dice: {D1}-{D2}    Cube: 2, owned by opponent")
    print(f"Eval level: 3T (XGRoller++ Checker = 360 trials, trunc_depth=5, "
          f"decision_ply=3, late_ply=2 from move 2)")
    print(f"Threads: {n_threads}\n")

    report(analyzer, "3T MONEY (Jacoby + Beaver)",
           jacoby=True, beaver=True)
    report(analyzer, "3T MATCH (player 4-away, opp 1-away)",
           away1=4, away2=1, is_crawford=False)

    print("Legend: cfEq = cubeful equity (per initial cube),  "
          "clEq = cubeless equity (no cube)")
    print("        Win/gW/bW = win / gammon-win / backgammon-win prob;  "
          "gL/bL = gammon-loss / backgammon-loss prob")
    print("        diff = cubeful equity loss vs best move")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    main(n_threads=args.threads)
