from pathlib import Path

import numpy as np
import pytest

from zsad.data import block_to_original, downsample_max, load_ucr, prepare, unwindow, windows

DATA = Path(__file__).parents[1] / "data" / "198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out"


def test_downsample_pads_on_the_left_to_a_multiple():
    values = np.arange(1, 26, dtype=float)  # 25 points -> 5 zeros of padding, 3 blocks
    labels = np.zeros(25, dtype=int)
    labels[12] = 1
    v, y, padding = downsample_max(values, labels)
    assert padding == 5
    np.testing.assert_array_equal(v, [5, 15, 25])  # blocks [pad*5, 1..5], [6..15], [16..25]
    np.testing.assert_array_equal(y, [0, 1, 0])  # value 13 (index 12) is in the second block


def test_downsample_adds_a_full_block_when_already_a_multiple():
    # Same quirk as the authors' code: 20 points get 10 zeros of padding, not 0.
    v, y, padding = downsample_max(np.arange(1, 21, dtype=float), np.zeros(20, dtype=int))
    assert padding == 10
    np.testing.assert_array_equal(v, [0, 10, 20])


def test_block_to_original_matches_downsample():
    rng = np.random.default_rng(0)
    values = rng.uniform(1, 2, size=137)  # positive, so zero padding never wins the max
    v, _, padding = downsample_max(values, np.zeros(137, dtype=int))
    for i in range(len(v)):
        first, last = block_to_original(i, 10, padding)
        assert last - first == 9
        assert v[i] == values[max(first, 0) : last + 1].max()
    assert block_to_original(len(v) - 1, 10, padding)[1] == len(values) - 1


def test_block_to_original_offset():
    # A segment that started at original index 50000, with padding 3: block 0 covers [49997, 50006].
    assert block_to_original(0, 10, 3, offset=50000) == (49997, 50006)


def test_load_ucr_tiltapb2():
    s = load_ucr(DATA)
    assert s.name == "tiltAPB2"
    assert len(s.values) == len(s.labels) == 128767
    assert s.values[0] == 4376.0  # the first row is data, not a header
    assert s.train_end == 50000
    assert (s.anomaly_start, s.anomaly_end) == (124159, 124984)
    assert s.labels.sum() == 124984 - 124159 + 1


def test_prepare_full_resolution():
    d = prepare(DATA)
    assert (len(d["fit"]), len(d["cal"]), len(d["test"])) == (42857, 7143, 78767)  # 6/7 of 50000
    train = np.concatenate([d["fit"], d["cal"]])  # scaled with the whole train part's mean and std
    assert abs(train.mean()) < 1e-9 and abs(train.std() - 1) < 1e-9
    assert len(d["labels"]) == len(d["test"]) and d["labels"].sum() == 826


def test_prepare_downsampled():
    d = prepare(DATA, factor=10)
    # Each part is max-pooled separately: 42857 -> 4286, 7143 -> 715, 78767 -> 7877 blocks.
    assert (len(d["fit"]), len(d["cal"]), len(d["test"])) == (4286, 715, 7877)
    assert d["labels"].sum() == 83  # test blocks 7416..7498 touch the anomaly


def test_windows_last_one_aligned_to_the_end():
    x = np.arange(1100.0)
    w = windows(x)
    assert w.shape == (3, 512) and w[2, 0] == 588 and w[2, -1] == 1099
    assert windows(np.arange(1024.0)).shape == (2, 512)  # exact multiple: no duplicated window
    with pytest.raises(ValueError):
        windows(np.arange(511.0))


@pytest.mark.parametrize("length", [512, 1024, 1100, 2000])
def test_windows_then_unwindow_is_identity(length):
    x = np.random.default_rng(0).normal(size=length)
    np.testing.assert_array_equal(unwindow(windows(x), length), x)


def test_unwindow_keeps_earlier_window_in_overlap():
    # Windows of 4 over 6 points: the second window covers points 2..5, of which only 4..5 are new.
    np.testing.assert_array_equal(unwindow(np.array([[1.0, 1, 1, 1], [2.0, 2, 2, 2]]), 6), [1, 1, 1, 1, 2, 2])
