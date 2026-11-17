# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""User settings for bgsage, currently the path to the GNUbg binary.

The GNUbg path is resolved in this order (first hit wins):

1. A path set in code with :func:`set_gnubg_path` (this process only), or
   passed per instance as ``GnuBgAnalyzer(gnubg_path=...)``.
2. The ``BGSAGE_GNUBG`` environment variable.
3. ``"gnubg_path"`` in the user settings file (see :func:`settings_file`),
   written by ``set_gnubg_path(path, persist=True)`` or by hand.
4. Auto-detection: the default Windows install
   (``C:\\Program Files (x86)\\gnubg\\gnubg-cli.exe``), then ``gnubg-cli`` or
   ``gnubg`` on PATH.

Example::

    from bgsage import settings
    settings.set_gnubg_path("/opt/homebrew/bin/gnubg", persist=True)
    settings.gnubg_path()          # -> "/opt/homebrew/bin/gnubg"
    settings.set_gnubg_path(None, persist=True)   # back to auto-detection
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

GNUBG_ENV_VAR = "BGSAGE_GNUBG"
SETTINGS_ENV_VAR = "BGSAGE_SETTINGS"
GNUBG_WINDOWS_DEFAULT = r"C:\Program Files (x86)\gnubg\gnubg-cli.exe"

_override: str | None = None


def settings_file() -> Path:
    """Location of the user settings file (JSON).

    ``BGSAGE_SETTINGS`` overrides it; otherwise
    ``%APPDATA%\\bgsage\\settings.json`` on Windows and
    ``$XDG_CONFIG_HOME/bgsage/settings.json`` (default ``~/.config``) elsewhere.
    """
    env = os.environ.get(SETTINGS_ENV_VAR)
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "bgsage" / "settings.json"


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
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def detect_gnubg_path() -> str | None:
    """Find an installed GNUbg binary without consulting any setting."""
    if os.path.isfile(GNUBG_WINDOWS_DEFAULT):
        return GNUBG_WINDOWS_DEFAULT
    for name in ("gnubg-cli", "gnubg"):
        found = shutil.which(name)
        if found:
            return found
    return None


def gnubg_path() -> str:
    """The GNUbg binary to run, resolved as described in the module docstring.

    When nothing is configured or found, returns the Windows default path so
    that running it fails with a FileNotFoundError naming a real location.
    """
    if _override:
        return _override
    env = os.environ.get(GNUBG_ENV_VAR)
    if env:
        return os.path.expanduser(env)
    saved = load_settings().get("gnubg_path")
    if isinstance(saved, str) and saved:
        return os.path.expanduser(saved)
    return detect_gnubg_path() or GNUBG_WINDOWS_DEFAULT


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
