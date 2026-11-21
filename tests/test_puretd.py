import os

import pytest

import neongammon as ng


@pytest.fixture(scope="module")
def puretd():
    library = os.environ.get("NEONGAMMON_TEST_PURETD_LIBRARY")
    model = os.environ.get("NEONGAMMON_TEST_PURETD_MODEL")
    if not library or not model:
        pytest.skip("PureTD integration fixture requires its native library and exported checkpoint")
    with ng.PureTDPolicy(library, model) as policy:
        yield policy


def test_missing_puretd_library_is_reported(tmp_path):
    path = tmp_path / "missing.dylib"
    with pytest.raises(FileNotFoundError, match="missing.dylib"):
        ng.PureTDPolicy(path, tmp_path / "model.bin")


@pytest.mark.parametrize("checker_plies", [-1, 2, 1.0, True])
def test_puretd_rejects_unsupported_search_depth(tmp_path, checker_plies):
    with pytest.raises(ValueError, match="checker_plies must be 0 or 1"):
        ng.PureTDPolicy(tmp_path / "library", tmp_path / "model", checker_plies=checker_plies)


@pytest.mark.parametrize("root_filter", [-0.01, float("inf"), float("nan"), 1e39, 1e-100, True, "0.075"])
def test_puretd_rejects_out_of_range_root_filter(tmp_path, root_filter):
    with pytest.raises(ValueError, match="root_filter"):
        ng.PureTDPolicy(tmp_path / "library", tmp_path / "model",
                        checker_plies=1, root_filter=root_filter)


def test_puretd_root_filter_requires_lookahead(tmp_path):
    with pytest.raises(ValueError, match="root_filter requires checker_plies=1"):
        ng.PureTDPolicy(tmp_path / "library", tmp_path / "model", root_filter=0.075)


def test_puretd_uses_native_root_filter_and_delegates_cube(puretd):
    board = [
        0, 0, -2, 4, 2, 0, 3, 2, 2, 0, 0, 0, 0,
        0, 0, -1, 0, 0, 2, -2, -2, -2, -2, -2, -2, 0,
    ]
    expected = [
        0, 0, -2, 4, 2, 0, 3, 2, 2, 0, 0, 0, 1,
        0, 0, -1, 1, 0, 0, -2, -2, -2, -2, -2, -2, 0,
    ]
    library = os.environ["NEONGAMMON_TEST_PURETD_LIBRARY"]
    model = os.environ["NEONGAMMON_TEST_PURETD_MODEL"]
    context = dict(cube_value=2, cube_owner="opponent", beaver=False)
    with (
        ng.PureTDPolicy(library, model, analysis=puretd.analysis, checker_plies=1) as full,
        ng.PureTDPolicy(library, model, analysis=puretd.analysis, checker_plies=1,
                        root_filter=0.075) as filtered,
    ):
        assert filtered.select_checker_move(board, 2, 6, **context) == expected
        assert full.select_checker_move(board, 2, 6, **context) != expected
        assert (filtered.cube_decision(board, **context).should_take
                == puretd.analysis.cube_decision(board, **context).should_take)


def test_implicit_jacoby_game_uses_filtered_puretd(puretd, monkeypatch):
    library = os.environ["NEONGAMMON_TEST_PURETD_LIBRARY"]
    model = os.environ["NEONGAMMON_TEST_PURETD_MODEL"]
    monkeypatch.setenv("NEONGAMMON_PURETD_LIBRARY", library)
    monkeypatch.setenv("NEONGAMMON_PURETD_MODEL", model)
    config = ng.GameConfig(max_cube_value=128)
    seed = 965005
    with ng.PureTDPolicy(library, model, checker_plies=1, root_filter=0.075) as filtered:
        expected = ng.simulate_game(seed=seed, config=config, players=(filtered, filtered))
    stage = ng.BgBotAnalyzer(parallel_threads=1)
    assert ng.simulate_game(seed=seed, config=config, players=(stage, stage)) != expected
    assert ng.simulate_game(seed=seed, config=config) == expected


@pytest.mark.parametrize("library_exists", [False, True])
def test_unavailable_puretd_falls_back_to_stage11(tmp_path, monkeypatch, library_exists):
    library = tmp_path / "unloadable.so"
    if library_exists:
        library.write_text("not a shared library")
    model = tmp_path / "model"
    model.touch()
    monkeypatch.setenv("NEONGAMMON_PURETD_LIBRARY", str(library))
    monkeypatch.setenv("NEONGAMMON_PURETD_MODEL", str(model))
    stage = ng.BgBotAnalyzer(parallel_threads=1)
    expected = ng.simulate_game(seed=42, players=(stage, stage))
    assert ng.simulate_game(seed=42) == expected


@pytest.mark.parametrize("config", [ng.GameConfig(jacoby=False), ng.GameConfig(match_length=5)])
def test_puretd_is_only_probed_for_jacoby_money_games(monkeypatch, config):
    from neongammon import puretd as adapter

    def unexpected_probe():
        pytest.fail("PureTD was probed outside Jacoby money play")

    monkeypatch.setattr(adapter, "_default_money_policy", unexpected_probe)
    stage = ng.BgBotAnalyzer(parallel_threads=1)
    assert ng.simulate_game(config=config) == ng.simulate_game(config=config, players=(stage, stage))


def test_puretd_one_reply_changes_checker_play_and_delegates_cube(puretd):
    board = [
        0, 0, 0, 0, 0, 0, 5, -2, 3, 0, 0, 0, -3,
        6, 0, 0, 0, -2, -2, -4, 0, -2, 0, 0, 1, 0,
    ]
    expected = [
        0, 0, 0, 0, 0, 1, 4, -2, 3, 0, 0, 0, -3,
        6, 0, 0, 0, -2, -2, -4, 1, -2, 0, 0, 0, 0,
    ]
    with ng.PureTDPolicy(
        os.environ["NEONGAMMON_TEST_PURETD_LIBRARY"],
        os.environ["NEONGAMMON_TEST_PURETD_MODEL"],
        checker_plies=1,
        analysis=puretd.analysis,
    ) as searched:
        assert searched.select_checker_move(board, 4, 1) == expected
        assert puretd.select_checker_move(board, 4, 1) != expected
        assert searched.cube_decision(board, beaver=False).should_double == (
            puretd.cube_decision(board, beaver=False).should_double
        )


def test_puretd_chooses_9_and_10_on_opening_43(puretd):
    board = puretd.select_checker_move(ng.STARTING_BOARD, 4, 3)
    assert board == [
        0, -2, 0, 0, 0, 0, 5, 0, 3, 1, 1, 0, -5,
        3, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
    ]
    assert board != puretd.analysis.checker_play(
        ng.STARTING_BOARD, 4, 3, beaver=False
    ).moves[0].board


def test_puretd_uses_stage11_outside_supported_cube_rules(puretd):
    board = ng.STARTING_BOARD
    for context in (
        {"away1": 5, "away2": 3},
        {"jacoby": False},
        {"beaver": True},
        {"cube_value": 2, "cube_owner": "player", "max_cube_value": 2},
    ):
        expected = puretd.analysis.checker_play(board, 4, 3, **{"beaver": False, **context})
        assert puretd.select_checker_move(board, 4, 3, **context) == expected.moves[0].board


def test_puretd_accepts_a_forced_pass(puretd):
    board = [0] * 26
    board[25] = 1
    board[19:25] = [-2] * 6
    assert puretd.select_checker_move(board, 6, 1) == board


def test_puretd_rejects_invalid_board_before_native_evaluation(puretd):
    board = ng.STARTING_BOARD.copy()
    board[25] = 16
    with pytest.raises(ValueError, match="checker counts"):
        puretd.select_checker_move(board, 4, 3)
