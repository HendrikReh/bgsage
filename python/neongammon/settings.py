# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""User settings for neongammon, currently the path to the GNUbg binary.

The GNUbg path is resolved in this order (first hit wins):

1. A path set in code with :func:`set_gnubg_path` (this process only), or
   passed per instance as ``GnuBgAnalyzer(gnubg_path=...)``.
2. The ``NEONGAMMON_GNUBG`` environment variable (``BGSAGE_GNUBG``, the
   name from before the rename, is still read when it is unset).
3. ``"gnubg_path"`` in the user settings file (see :func:`settings_file`),
   written by ``set_gnubg_path(path, persist=True)`` or by hand.
4. Auto-detection: ``gnubg`` on PATH, then the Homebrew locations
   (``/opt/homebrew/bin/gnubg``, ``/usr/local/bin/gnubg``).

Example::

    from neongammon import settings
    settings.set_gnubg_path("/opt/homebrew/bin/gnubg", persist=True)
    settings.gnubg_path()          # -> "/opt/homebrew/bin/gnubg"
    settings.set_gnubg_path(None, persist=True)   # back to auto-detection
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

GNUBG_ENV_VAR = "NEONGAMMON_GNUBG"
SETTINGS_ENV_VAR = "NEONGAMMON_SETTINGS"
# The names before the project was renamed from bgsage (2026-11-20). Each is
# read only when its new counterpart is absent.
LEGACY_GNUBG_ENV_VAR = "BGSAGE_GNUBG"
LEGACY_SETTINGS_ENV_VAR = "BGSAGE_SETTINGS"
LEGACY_SETTINGS_DIR = "bgsage"
GNUBG_DEFAULT = "gnubg"
GNUBG_HOMEBREW_PATHS = ("/opt/homebrew/bin/gnubg", "/usr/local/bin/gnubg")

_override: str | None = None


def _env(name: str, legacy: str) -> str | None:
    return os.environ.get(name) or os.environ.get(legacy) or None


def _config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _settings_path(*, for_write: bool) -> Path:
    env = _env(SETTINGS_ENV_VAR, LEGACY_SETTINGS_ENV_VAR)
    if env:
        return Path(env).expanduser()
    path = _config_dir() / "neongammon" / "settings.json"
    legacy = _config_dir() / LEGACY_SETTINGS_DIR / "settings.json"
    if not for_write and not path.exists() and legacy.exists():
        return legacy
    return path


def settings_file() -> Path:
    """Location of the user settings file (JSON) that is read.

    ``NEONGAMMON_SETTINGS`` (or the older ``BGSAGE_SETTINGS``) overrides it;
    otherwise ``$XDG_CONFIG_HOME/neongammon/settings.json`` (default
    ``~/.config``). When that file does not exist but one from before the
    rename does (``.../bgsage/settings.json``), the old one is read, so a
    saved GNUbg path survives the rename; :func:`save_settings` always writes
    the new file.
    """
    return _settings_path(for_write=False)


def load_settings() -> dict:
    """Read the settings file; an absent or unreadable file gives ``{}``."""
    try:
        with open(settings_file(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data: dict) -> Path:
    """Write ``data`` as the settings file (creating its folder); returns its path."""
    path = _settings_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def detect_gnubg_path() -> str | None:
    """Find an installed GNUbg binary without consulting any setting."""
    found = shutil.which(GNUBG_DEFAULT)
    if found:
        return found
    for path in GNUBG_HOMEBREW_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def gnubg_path() -> str:
    """The GNUbg binary to run, resolved as described in the module docstring.

    When nothing is configured or found, returns ``"gnubg"``, so running it
    fails with a FileNotFoundError naming the binary.
    """
    if _override:
        return _override
    env = _env(GNUBG_ENV_VAR, LEGACY_GNUBG_ENV_VAR)
    if env:
        return os.path.expanduser(env)
    saved = load_settings().get("gnubg_path")
    if isinstance(saved, str) and saved:
        return os.path.expanduser(saved)
    return detect_gnubg_path() or GNUBG_DEFAULT


def set_gnubg_path(path: str | os.PathLike | None, *, persist: bool = False) -> None:
    """Set the GNUbg binary for this process; ``None`` clears it.

    With ``persist=True`` the value is also written to (or, for ``None``,
    removed from) the user settings file so later sessions use it.
    """
    global _override
    _override = os.fspath(path) if path is not None else None
    if persist:
        data = load_settings()
        if _override is None:
            data.pop("gnubg_path", None)
        else:
            data["gnubg_path"] = _override
        save_settings(data)
