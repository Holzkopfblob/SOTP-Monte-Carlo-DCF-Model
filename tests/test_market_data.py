"""ERP-Abruf: Parser für Damodarans Startseite (ohne Netz)."""
import pytest

from infrastructure.market_data import parse_damodaran_erp


def test_parse_damodaran_erp():
    html = ("<p><b>Implied ERP on November 1, 2026</b> = 4.41% (Trailing 12 month, with adjusted payout); "
            "3.79% (Trailing 12 month cash yield)</p><p>Implied ERP in previous month = 4.20% …</p>")
    assert parse_damodaran_erp(html) == (pytest.approx(0.0441), "2026-11-01")
    with pytest.raises(ValueError):
        parse_damodaran_erp("<p>keine Zahl</p>")
