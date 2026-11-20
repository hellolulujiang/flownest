"""The small rules of the SWORD crosswalk that need no data: how ratios and covers are written, and how the issue codes
give match_confidence. The matching itself is compared with the release on six SWORD subnetworks by
release_checks/check_sword_python_against_release.py (README, "Verification and limits")."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('pyogrio')
pytest.importorskip('shapely')
from flownest.sword import confidence as CF            # noqa: E402
from flownest.sword.reaches import stored, cover_text  # noqa: E402


def test_stored_keeps_six_significant_digits():
    v = stored([0.123456789, 12345.6789, 0.0, np.nan, np.inf, -2.5e-7])
    assert v[0] == 0.123457 and v[1] == 12345.7 and v[2] == 0.0 and np.isnan(v[3]) and np.isinf(v[4]) and v[5] == -2.5e-7


def test_cover_text_never_crosses_80_percent():
    assert cover_text(0.75) == '0.75'
    assert cover_text(0.7999999) == '0.7999999'        # 6 digits would read 0.8, on the other side of the rule
    assert cover_text(0.8) == '0.8'
    assert cover_text(0.912345678) == '0.912346'


def _reach(issue='', note='', nf='', cover=0.9, check='agrees', placed=10, far=0, orient='along_flow'):
    return dict(issue=issue, area_note=note, not_found_reason=nf, cover=cover, area_check=check,
                nodes_placed=placed, nodes_far_from_pixel=far, orientation=orient)


def test_reach_confidence_tiers():
    R = pd.DataFrame([
        _reach(),                                                              # all four pieces of evidence
        _reach(cover=0.79),                                                    # cover short of 80%
        _reach(issue='upstream_area_disagrees', check='disagrees'),            # not explained
        _reach(issue='upstream_area_disagrees', check='disagrees', note='side_channel'),
        _reach(issue='matched_by_upstream_area; most_nodes_far_from_pixel'),   # low listed before medium
        _reach(issue='not_matched', nf='far_from_path', cover=np.nan, check='', placed=0),
    ])
    t, r = CF.reach_confidence(R)
    assert list(t) == ['high', 'medium', 'low', 'medium', 'low', 'not_found']
    assert r[0] == CF.HIGH_REASON
    assert r[1] == 'cover_below_80pct'
    assert r[2] == 'upstream_area_disagrees_unexplained'
    assert r[3] == 'upstream_area_disagrees_explained'
    assert r[4] == 'most_nodes_far_from_pixel; matched_by_upstream_area'
    assert r[5] == 'far_from_path'


def test_node_confidence_takes_the_worse_tier():
    N = pd.DataFrame({'sword_reach_id': [1, 1, 2, 3], 'issue': ['', 'far_from_pixel', '', 'not_placed']})
    tier = pd.Series({1: 'high', 2: 'not_found', 3: 'medium'})
    t, r = CF.node_confidence(N, tier)
    assert list(t) == ['high', 'low', 'low', 'not_found']
    assert r[0] == 'node_near_pixel; reach_high'
    assert r[1] == 'far_from_pixel; reach_high'
    assert r[2] == 'reach_not_found'
    assert r[3] == 'not_placed; reach_medium'


def test_unknown_code_stops():
    with pytest.raises(ValueError):
        CF.reach_confidence(pd.DataFrame([_reach(issue='some_new_code')]))


def test_same_area_window_reads_across_180():
    from flownest.sword.finish import read_across_180

    class Band:                                         # a grid of 10 columns whose value is the column number
        width = 10

        def read(self, c, r, w, h):
            return np.tile(np.arange(c, c + w, dtype=np.float32), (h, 1))

    assert read_across_180(Band(), -2, 0, 5, 2)[0].tolist() == [8, 9, 0, 1, 2]     # west of column 0
    assert read_across_180(Band(), 8, 0, 4, 1)[0].tolist() == [8, 9, 0, 1]         # past the last column
    assert read_across_180(Band(), 3, 0, 3, 1)[0].tolist() == [3, 4, 5]
