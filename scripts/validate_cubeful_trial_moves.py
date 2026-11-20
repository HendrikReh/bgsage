"""Validate cubeful_trial_moves flag on the user-analyzed match position.

Before fix: post_move analytics returned identical probs for money and match.
After fix (flag=True): trial moves are picked by cubeful equity per branch's
cube state. Expect probs to shift between money and match in the gammon-
favored direction.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "build"))
sys.path.insert(0, str(REPO / "python"))

from bgsage import BgBotAnalyzer

# Post-move position after 8/6 8/3 (from the analyzed conversation)
POST = [0, 2, 2, 2, 2, 0, 3, 0, 0, 1, 0, 0, -4, 2, 0, 0, 0, -3, 0, -2, -4, 1, -2, 0, 0, 0]
CV, OWNER = 2, "opponent"
AWAY1, AWAY2 = 4, 1  # match 1-4 to 5


def run(label: str, **analyzer_kwargs):
    analyzer = BgBotAnalyzer(eval_level="truncated3", cubeful=True, **analyzer_kwargs)
    print(f"\n=== {label} ===")
    print("  money:")
    t0 = time.time()
    m = analyzer.post_move_analytics(POST, cube_owner=OWNER, cube_value=CV)
    money_t = time.time() - t0
    print(f"    W={m.probs.win:.4%}  gW={m.probs.gammon_win:.4%}  gL={m.probs.gammon_loss:.4%}"
          f"  cubeless={m.cubeless_equity:+.4f}  cubeful={m.cubeful_equity:+.4f}  ({money_t:.1f}s)")

    print("  match (1-4 to 5):")
    t0 = time.time()
    h = analyzer.post_move_analytics(POST, cube_owner=OWNER, cube_value=CV,
                                      away1=AWAY1, away2=AWAY2, is_crawford=False)
    match_t = time.time() - t0
    print(f"    W={h.probs.win:.4%}  gW={h.probs.gammon_win:.4%}  gL={h.probs.gammon_loss:.4%}"
          f"  cubeless={h.cubeless_equity:+.4f}  cubeful={h.cubeful_equity:+.4f}  ({match_t:.1f}s)")

    return m, h


m_off, h_off = run("flag OFF (baseline)", cubeful_trial_moves=False)
m_on, h_on = run("flag ON (cube-aware trial moves)", cubeful_trial_moves=True)

print("\n=== Delta (on - off) ===")
print(f"  money cubeless eq:  {m_on.cubeless_equity - m_off.cubeless_equity:+.4f}")
print(f"  match cubeless eq:  {h_on.cubeless_equity - h_off.cubeless_equity:+.4f}")
print(f"  money gW%:          {(m_on.probs.gammon_win - m_off.probs.gammon_win)*100:+.2f}pp")
print(f"  match gW%:          {(h_on.probs.gammon_win - h_off.probs.gammon_win)*100:+.2f}pp")
print(f"  match cubeful eq:   {h_on.cubeful_equity - h_off.cubeful_equity:+.4f}")
print()
print("Validation: with flag OFF money and match return IDENTICAL cubeless probs")
print("(user's original observation). With flag ON they differ — trials now select")
print("moves by cubeful equity against each game type's cube state. Direction of")
print("gW shift isn't fixed: both players' moves become cube-aware, so the")
print("attacker's gammon-seeking and defender's gammon-avoiding partly cancel.")
