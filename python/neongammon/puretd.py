# SPDX-License-Identifier: MPL-2.0
"""Opt-in PureTD checker play for Jacoby money games."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from threading import Lock
from weakref import finalize

from .analyzer import BgBotAnalyzer, _validate_cube_limit
from .board import possible_moves
from .types import CubeDecisionResult


# PureTD's C ABI 7, declared in c_engine/bg_engine.h and c_api/puretd.h.
class _Board(ctypes.Structure):
    _fields_ = [
        ("points", ctypes.c_int * 24),
        ("bar", ctypes.c_int * 2),
        ("off", ctypes.c_int * 2),
        ("turn", ctypes.c_int),
    ]


class _Cube(ctypes.Structure):
    _fields_ = [
        ("owner", ctypes.c_int),
        ("value", ctypes.c_int),
        ("jacoby", ctypes.c_int),
    ]


class _RootSearch(ctypes.Structure):
    _fields_ = [("root_thresh", ctypes.c_float)]


_MAX_C_FLOAT = float.fromhex("0x1.fffffep127")
_CUBE_OWNER = {"centered": 0, "player": 1, "opponent": 2}


def _default_money_policy() -> PureTDPolicy | None:
    """Load the tested policy for an implicit Jacoby money game."""
    library = os.environ.get("NEONGAMMON_PURETD_LIBRARY")
    model = os.environ.get("NEONGAMMON_PURETD_MODEL")
    if library is None and model is None:
        native = Path(__file__).resolve().parents[2] / "build" / "puretd"
        library = native / "libpuretd_cblas.so"
        model = native / "cubeful.bin"
    if not library or not model:
        return None
    try:
        library_path = Path(library).expanduser()
        model_path = Path(model).expanduser()
        if not library_path.is_file() or not model_path.is_file():
            return None
        return PureTDPolicy(library_path, model_path, checker_plies=1, root_filter=0.075)
    except (OSError, AttributeError, ValueError):
        # The optional library, ABI, search symbol or weights could not be loaded.
        return None


def _load_library(path):
    lib = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    lib.bgc_abi_version.argtypes = []
    lib.bgc_abi_version.restype = ctypes.c_int
    if lib.bgc_abi_version() != 7:
        raise ValueError("PureTDPolicy requires PureTD C ABI version 7")

    lib.bgc_struct_sizes.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    lib.bgc_struct_sizes.restype = ctypes.c_int
    sizes = (ctypes.c_size_t * 3)()
    if (lib.bgc_struct_sizes(sizes, len(sizes)) < len(sizes)
            or sizes[0] != ctypes.sizeof(_Board) or sizes[2] != ctypes.sizeof(_Cube)):
        raise ValueError("PureTD board or cube ABI layout does not match")

    lib.bgc_open_ex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.bgc_open_ex.restype = ctypes.c_void_p
    lib.bgc_kind.argtypes = [ctypes.c_void_p]
    lib.bgc_kind.restype = ctypes.c_int
    lib.bgc_close.argtypes = [ctypes.c_void_p]
    lib.bgc_close.restype = None
    lib.bgc_best_move.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_Board), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(_Cube), ctypes.c_int, ctypes.POINTER(_Board),
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.bgc_best_move.restype = ctypes.c_int
    return lib


class PureTDPolicy:
    """Choose money-game moves with PureTD and use an analyzer for the cube.

    Pass a local PureTD ABI-7 library and a cubeful checkpoint exported with
    PureTD's ``export_weights.py``. ``checker_plies`` counts searched replies
    after each candidate move (0 or 1). With one reply, a positive
    ``root_filter`` limits search to candidates within that equity of the
    best static move and includes game-ending moves. The default of zero
    uses full-width search. ``analysis`` provides full analysis, cube
    decisions, and checker play outside Jacoby money games without beavers.
    The analyzer also handles a cube that has reached its configured limit.
    """

    def __init__(
        self,
        library_path: str | os.PathLike[str],
        model_path: str | os.PathLike[str],
        *,
        analysis: BgBotAnalyzer | None = None,
        checker_plies: int = 0,
        root_filter: float = 0.0,
    ):
        if type(checker_plies) is not int or checker_plies not in (0, 1):
            raise ValueError("checker_plies must be 0 or 1")
        if (isinstance(root_filter, bool) or not isinstance(root_filter, (int, float))
                or not 0 <= root_filter <= _MAX_C_FLOAT):
            raise ValueError("root_filter must be a finite nonnegative native float")
        search = _RootSearch(root_filter)
        if root_filter and search.root_thresh == 0:
            raise ValueError("root_filter is too small to represent in the native library")
        if root_filter and checker_plies != 1:
            raise ValueError("a positive root_filter requires checker_plies=1")
        self._checker_plies = checker_plies
        self._lib = _load_library(library_path)
        configure_search = None
        if root_filter:
            configure_search = getattr(self._lib, "bgc_set_search", None)
            if configure_search is None:
                raise ValueError("PureTD library does not support a positive root_filter")
            configure_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RootSearch)]
            configure_search.restype = ctypes.c_int
        model = Path(model_path).resolve(strict=True)
        error = ctypes.create_string_buffer(1024)
        handle = self._lib.bgc_open_ex(os.fsencode(model), error, len(error))
        if not handle:
            raise ValueError(f"Could not open PureTD model {model}: {error.value.decode(errors='replace')}")
        self._release = finalize(self, self._lib.bgc_close, handle)
        if self._lib.bgc_kind(handle) != 2:  # BGC_KIND_CUBEFUL
            self._release()
            raise ValueError("PureTDPolicy requires a cubeful money model")
        if configure_search is not None and configure_search(handle, ctypes.byref(search)) != 0:
            self._release()
            raise ValueError(f"PureTD rejected root_filter={root_filter!r}")
        self._handle = handle
        self._lock = Lock()
        self.analysis = analysis if analysis is not None else BgBotAnalyzer(
            eval_level="2ply", parallel_threads=1
        )

    def close(self) -> None:
        """Release the loaded PureTD model."""
        with self._lock:
            self._release()

    def __enter__(self) -> PureTDPolicy:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def cube_decision(self, *args, **kwargs) -> CubeDecisionResult:
        """Evaluate the cube with the configured analyzer."""
        return self.analysis.cube_decision(*args, **kwargs)

    def select_checker_move(
        self,
        board: list[int],
        die1: int,
        die2: int,
        cube_value: int = 1,
        cube_owner: str = "centered",
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = False,
        max_cube_value: int = 0,
    ) -> list[int]:
        """Return a legal post-move board in the mover's perspective."""
        if len(board) != 26 or any(type(checkers) is not int for checkers in board):
            raise ValueError("board must contain 26 integers")
        points = board[1:25]
        mover_off = 15 - board[25] - sum(p for p in points if p > 0)
        opponent_off = 15 - board[0] + sum(p for p in points if p < 0)
        if min(board[0], board[25], mover_off, opponent_off) < 0:
            raise ValueError("invalid checker counts")
        if max(mover_off, opponent_off) == 15:
            raise ValueError("board must be nonterminal")
        if any(type(die) is not int or not 1 <= die <= 6 for die in (die1, die2)):
            raise ValueError("dice must be integers in [1, 6]")
        if (type(cube_value) is not int or not 1 <= cube_value <= 1 << 30
                or cube_value & (cube_value - 1)):
            raise ValueError("cube_value must be a power of two in [1, 2**30]")
        _validate_cube_limit(cube_value, max_cube_value)
        if cube_owner not in _CUBE_OWNER:
            raise ValueError("invalid cube_owner")

        with self._lock:
            if not self._release.alive:
                raise RuntimeError("PureTD policy is closed")
            if not (away1 or away2 or is_crawford or beaver or not jacoby
                    or cube_value == max_cube_value):
                legal = possible_moves(board, die1, die2)
                if len(legal) == 1:
                    return legal[0]
                state = _Board((ctypes.c_int * 24)(*points), (board[25], board[0]),
                               (mover_off, opponent_off), 0)
                cube = _Cube(_CUBE_OWNER[cube_owner], cube_value, 1)
                successor = _Board()
                index = self._lib.bgc_best_move(self._handle, ctypes.byref(state),
                                                die1, die2, ctypes.byref(cube),
                                                self._checker_plies,
                                                ctypes.byref(successor), None, None)
                if index < 0:
                    raise RuntimeError("PureTD could not select a checker move")
                selected = [successor.bar[1], *successor.points, successor.bar[0]]
                if selected not in legal:
                    raise RuntimeError("PureTD returned an illegal checker move")
                return selected

        return self.analysis.checker_play(
            board, die1, die2, cube_value, cube_owner, away1=away1, away2=away2,
            is_crawford=is_crawford, jacoby=jacoby, beaver=beaver,
            max_cube_value=max_cube_value,
        ).moves[0].board
