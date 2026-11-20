# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Tests for bgsage.settings (GNUbg path resolution and the settings file)."""

import json
import os
import sys

import pytest

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, 'build'))
sys.path.insert(0, os.path.join(project_dir, 'python'))

from bgsage import settings  # noqa: E402
from bgsage import gnubg  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point the settings file at a temp dir and clear every override."""
    monkeypatch.setenv(settings.SETTINGS_ENV_VAR, str(tmp_path / 'settings.json'))
    monkeypatch.delenv(settings.GNUBG_ENV_VAR, raising=False)
    monkeypatch.setattr(settings, '_override', None)
    monkeypatch.setattr(settings, 'detect_gnubg_path', lambda: '/detected/gnubg')
    yield tmp_path


def test_auto_detection_is_the_fallback():
    assert settings.gnubg_path() == '/detected/gnubg'


def test_nothing_found_falls_back_to_plain_gnubg(monkeypatch):
    monkeypatch.setattr(settings, 'detect_gnubg_path', lambda: None)
    assert settings.gnubg_path() == settings.GNUBG_DEFAULT


def test_settings_file_beats_detection(isolated):
    (isolated / 'settings.json').write_text(json.dumps({'gnubg_path': '/from/file'}))
    assert settings.gnubg_path() == '/from/file'


def test_env_var_beats_settings_file(isolated, monkeypatch):
    (isolated / 'settings.json').write_text(json.dumps({'gnubg_path': '/from/file'}))
    monkeypatch.setenv(settings.GNUBG_ENV_VAR, '/from/env')
    assert settings.gnubg_path() == '/from/env'


def test_code_setting_beats_env_var(monkeypatch):
    monkeypatch.setenv(settings.GNUBG_ENV_VAR, '/from/env')
    settings.set_gnubg_path('/from/code')
    assert settings.gnubg_path() == '/from/code'
    settings.set_gnubg_path(None)
    assert settings.gnubg_path() == '/from/env'


def test_persist_writes_and_clears_the_file(isolated):
    settings.set_gnubg_path('/saved/gnubg', persist=True)
    path = isolated / 'settings.json'
    assert json.loads(path.read_text()) == {'gnubg_path': '/saved/gnubg'}

    settings.set_gnubg_path(None)          # process override gone, file stays
    assert settings.gnubg_path() == '/saved/gnubg'

    settings.set_gnubg_path(None, persist=True)
    assert json.loads(path.read_text()) == {}
    assert settings.gnubg_path() == '/detected/gnubg'


def test_persist_keeps_other_keys(isolated):
    path = isolated / 'settings.json'
    path.write_text(json.dumps({'other': 1}))
    settings.set_gnubg_path('/x', persist=True)
    assert json.loads(path.read_text()) == {'gnubg_path': '/x', 'other': 1}


def test_corrupt_settings_file_is_ignored(isolated):
    (isolated / 'settings.json').write_text('{not json')
    assert settings.gnubg_path() == '/detected/gnubg'


def test_gnubg_module_reports_resolved_path():
    settings.set_gnubg_path('/from/code')
    assert gnubg.GNUBG_CLI == '/from/code'


def test_run_gnubg_uses_explicit_then_setting(monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stdout = ''
        stderr = ''

    def fake_run(argv, **kwargs):
        calls.append(argv[0])
        return Done()

    monkeypatch.setattr(gnubg.subprocess, 'run', fake_run)
    settings.set_gnubg_path('/from/code')
    gnubg._run_gnubg('quit')
    gnubg._run_gnubg('quit', gnubg_path='/per/call')
    assert calls == ['/from/code', '/per/call']
    assert gnubg.GnuBgAnalyzer(gnubg_path='/per/analyzer').gnubg_path == '/per/analyzer'
