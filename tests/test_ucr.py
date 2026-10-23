from collections import Counter

import numpy as np
import pytest

from zsad.ucr import (
    STUDY_IDS, archive_members, consistent_parts, load_archive, load_tsb, null_pairing, paper_parts,
    source_group, with_context,
)

pytestmark = pytest.mark.skipif(not __import__("pathlib").Path("data/ucr").exists(), reason="data not downloaded")


@pytest.fixture(scope="module")
def members():
    return archive_members()


def test_26_source_groups(members):
    names = {i: members[i].split("/")[-1].split("_")[3] for i in STUDY_IDS}
    counts = Counter(source_group(n) for n in names.values())
    assert len(STUDY_IDS) == 248 and len(counts) == 26
    assert counts["InternalBleeding"] == 26 and counts["EPG"] == 25 and counts["tilt"] == 14


def test_pairing_crosses_groups(members):
    groups = {i: source_group(members[i].split("/")[-1].split("_")[3]) for i in STUDY_IDS}
    pairing = null_pairing(groups)
    assert pairing == null_pairing(groups)  # fixed
    assert all(groups[i] != groups[j] for i, j in pairing.items())


def test_tiltapb2_in_both_configurations(members):
    a, t = load_archive(198, members), load_tsb(198)
    # Archive: 1-based inclusive 124159..124985 -> 0-based [124158, 124984].
    assert np.flatnonzero(a["labels"])[[0, -1]].tolist() == [124158, 124984]
    # TSB-UAD read without header=None loses its first row: values shift by one, and its
    # [begin, end) labels [124159, 124984] become [124158, 124983].
    np.testing.assert_array_equal(t["values"], a["values"][1:])
    assert np.flatnonzero(t["labels"])[[0, -1]].tolist() == [124158, 124983]
    c = consistent_parts(a)
    assert c["factor"] == 10 and c["labels"].sum() > 0
    p = paper_parts(t, 0.6 / 0.7)
    assert len(p["test"]) % 512 == 0 and len(p["test"]) == len(p["labels"])  # the tail is not scored


def test_with_context():
    history, part = np.arange(1000.0), np.arange(1000.0, 1100.0)
    scored = with_context(lambda x: x * 2, history, part)  # the score function sees 512 points
    np.testing.assert_array_equal(scored, part * 2)
    with pytest.raises(ValueError):
        with_context(lambda x: x, history[:10], part)
