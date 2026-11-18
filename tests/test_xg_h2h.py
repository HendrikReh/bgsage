# SPDX-License-Identifier: MPL-2.0
"""scripts/xg_h2h.py: the replay-to-a-fixed-point games must be the games a
live opponent would have played, and a round's XG files must verify."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "scripts", ROOT / "build", ROOT / "python"):
    sys.path.insert(0, str(p))

xg_h2h = pytest.importorskip("xg_h2h")


class LiveOpponent(xg_h2h.XgAnswers):
    """Answers every opponent decision on the spot, like a live opponent."""

    def __init__(self, oracle):
        self.rows, self.conflicts, self.oracle = {}, [], oracle

    def get(self, key):
        kind, board, cv, owner, *dice = key.split("|")
        board, cv = [int(x) for x in board.split(",")], int(cv)
        if kind == "C":
            return self.oracle.cube(board, cv, owner)
        d = dice[0]
        return self.oracle.checker(board, cv, owner, int(d[0]), int(d[1]))


def test_fixed_point_equals_live_play(tmp_path):
    run_dir = tmp_path / "sim"
    xg_h2h.main(["init", "--dir", str(run_dir), "--pairs", "2", "--seed", "3",
                 "--sage-level", "1ply", "--standin", "2ply"])
    xg_h2h.main(["round", "--dir", str(run_dir), "--simulate-xg"])
    run = xg_h2h.Run(run_dir)
    replayed = run.play_all()
    assert all(r["status"] == "done" for r in replayed.values())

    live_xg = LiveOpponent(xg_h2h.SageOracle("2ply", tmp_path / "live_cache.jsonl"))
    for gid, pair, seat in run.games:
        live = xg_h2h.play_game(run.cfg["seed"], pair, seat, run.sage, live_xg)
        assert live["status"] == "done"
        strip = lambda evs: [{k: v for k, v in e.items() if k != "src"} for e in evs]  # noqa: E731
        assert strip(live["events"]) == strip(replayed[gid]["events"]), gid
        assert live["result"] == replayed[gid]["result"]
        # every opponent decision of a finished game is a certified answer
        assert not any(e["src"] == "pred" for e in replayed[gid]["events"])


def test_duplicate_pairs_share_dice_and_opening():
    for pair in range(20):
        d1, d2 = xg_h2h.dice_for(5, pair, 0)
        assert d1 != d2
        assert [xg_h2h.dice_for(5, pair, t) for t in range(30)] == \
            [xg_h2h.dice_for(5, pair, t) for t in range(30)]


def test_round_writes_verified_shards_and_waits_for_xg(tmp_path):
    template = next((p for p in xg_h2h.DEFAULT_TEMPLATES if p.exists()), None)
    if template is None:
        pytest.skip("no XG template .xg on this machine")
    run_dir = tmp_path / "xg"
    xg_h2h.main(["init", "--dir", str(run_dir), "--pairs", "1", "--sage-level", "1ply"])
    xg_h2h.main(["round", "--dir", str(run_dir)])          # writes + verifies round_001
    shards = list((run_dir / "round_001").glob("shard_*.xg"))
    assert shards
    with pytest.raises(SystemExit, match="no XG answer yet"):
        xg_h2h.main(["round", "--dir", str(run_dir)])
