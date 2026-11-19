from presentation.format import eur, mio, num, pct


def test_german_formats():
    assert num(1234567.891, 2) == "1.234.567,89"
    assert num(-0.5, 1, signed=True) == "-0,5"
    assert num(3, 0, signed=True) == "+3"
    assert num(float("nan")) == num(None) == "–"
    assert pct(0.1234) == "12,3 %"
    assert pct(-0.25, 0, signed=True) == "-25 %"
    assert eur(1500) == "1.500 €"
    assert mio(9_999) == "9.999 Mio."
    assert mio(162_393.9) == "162,4 Mrd."
