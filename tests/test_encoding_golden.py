# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""
Golden regression test for the encoder's hand-optimised feature functions.

hitting_shots, double_hitting_shots, compute_piploss, moment2 and backbone are
bit-twiddled for speed; their hand-written unit tests (test_extended_inputs*.py)
cover small constructed positions. This test pins their values on 1,000
realistic positions (money-benchmark boards and their legal candidate moves,
900 contact positions with blots plus 100 races), as computed by the original
straightforward implementations. Any change to those functions must keep every
value identical.
"""

import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32' and os.path.isdir(build_dir):
    os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'python'))

import bgbot_cpp  # noqa: E402
from bgsage import flip_board  # noqa: E402

with open(os.path.join(script_dir, 'data', 'encoding_golden.json')) as f:
    GOLDEN = json.load(f)['positions']


def _mismatches(name, fn):
    bad = []
    for row in GOLDEN:
        b = row['board']
        got = [fn(b), fn(flip_board(b))]
        if got != row[name]:
            bad.append((b, row[name], got))
    return bad


def test_golden_has_positions():
    assert len(GOLDEN) == 1000


def test_hitting_shots_golden():
    assert _mismatches('shots', bgbot_cpp.hitting_shots)[:5] == []


def test_double_hitting_shots_golden():
    assert _mismatches('double_shots', bgbot_cpp.double_hitting_shots)[:5] == []


def test_piploss_golden():
    assert _mismatches('piploss', bgbot_cpp.compute_piploss)[:5] == []


def test_extended_inputs_golden():
    """Shots, piploss, moment2 and backbone inputs for both sides."""
    bad = []
    for row in GOLDEN:
        e = bgbot_cpp.compute_extended_contact_inputs(row['board'])
        for idx, want in row['inputs'].items():
            if abs(e[int(idx)] - want) > 1e-6:
                bad.append((row['board'], idx, want, e[int(idx)]))
    assert bad[:5] == []
