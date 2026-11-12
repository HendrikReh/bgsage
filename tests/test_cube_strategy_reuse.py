from unittest.mock import patch

import bgbot_cpp
import bgsage
import pytest


@pytest.mark.parametrize("level,binding", [
    ("1ply", "evaluate_cube_decision_unified"),
    ("2ply", "cube_decision_nply_unified"),
])
def test_cube_reuses_strategy_with_identical_results(level, binding):
    analyzer = bgsage.BgBotAnalyzer(eval_level=level, parallel_threads=2)
    native = getattr(bgbot_cpp, binding)
    def compare(*args, **kwargs):
        result = native(*args, **kwargs)
        reference = {key: value for key, value in kwargs.items() if key != "strategy"}
        assert result == native(*args, **reference)
        return result

    with patch.object(bgbot_cpp, binding, side_effect=compare) as evaluate:
        for away1, away2 in [(0, 0), (5, 4), (1, 3)]:
            analyzer.cube_action(list(bgsage.STARTING_BOARD), away1=away1, away2=away2)
    strategies = [call.kwargs.get("strategy") for call in evaluate.call_args_list]
    assert strategies[0] is not None
    assert all(strategy is strategies[0] for strategy in strategies)


def test_one_ply_checker_conversion_uses_no_pool():
    analyzer = bgsage.BgBotAnalyzer(eval_level="1ply", parallel_threads=2)
    with patch("bgsage.analyzer.ThreadPoolExecutor") as executor:
        # A mock is sufficient to detect accidental pool creation before any
        # conversion. The successful path evaluates the real opening position.
        executor.side_effect = AssertionError("1-ply started a candidate pool")
        result = analyzer.checker_play(list(bgsage.STARTING_BOARD), 3, 1)
    assert result.moves and all(move.eval_level == "1-ply" for move in result.moves)
