import numpy as np
import pytest

from zsad.windows import make_windows, stitch, window_starts


def test_window_starts():
    assert window_starts(512) == [0]
    assert window_starts(1024) == [0, 512]  # exact multiple: no duplicated last window
    assert window_starts(1100) == [0, 512, 588]  # last window aligned to the end


def test_window_starts_too_short():
    with pytest.raises(ValueError):
        window_starts(511)


@pytest.mark.parametrize("length", [512, 1024, 1100, 2000])
def test_make_windows_then_stitch_is_identity(length):
    x = np.random.default_rng(0).normal(size=length)
    windows, starts = make_windows(x)
    assert windows.shape == (len(starts), 512)
    np.testing.assert_array_equal(stitch(windows, starts, length), x)


def test_stitch_keeps_earlier_window_in_overlap():
    # Windows of 4 over 6 points: starts [0, 2], overlap at indices 2 and 3.
    per_window = np.array([[1.0, 1, 1, 1], [2.0, 2, 2, 2]])
    np.testing.assert_array_equal(stitch(per_window, [0, 2], 6), [1, 1, 1, 1, 2, 2])
