import numpy as np

from presentation.charts.risk import p1_p99_range


def test_p1_p99_range_cuts_tails_but_keeps_reference_lines():
    v = np.r_[np.linspace(100, 200, 10_000), 10.0, 1_000.0]          # two extreme outliers
    lo, hi = p1_p99_range(v)
    assert 90 < lo < 105 and 195 < hi < 210                           # outliers no longer stretch the axis
    lo, hi = p1_p99_range(v, 280.0)                                    # price above P99 stays visible
    assert hi > 280 and lo < 105
