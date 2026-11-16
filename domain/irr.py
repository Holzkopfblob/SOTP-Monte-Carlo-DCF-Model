"""
Implizite Rendite (IRR) je Szenario: die Rate r, bei der der Unternehmenswert aus den simulierten Cashflows
genau den Marktpreis ergibt (v3-Plan, Phase 1).

Wer eine Aktie dauerhaft hält, verdient langfristig diese Rendite der Cashflows zum Kaufkurs, unabhängig von
späteren Neubewertungen. Sie ersetzt „Wert bei CAPM-WACC vs. Kurs“ als Leitgröße: kein WACC-Urteil nötig,
das Risiko steckt in der Streuung über die Szenarien.

Gelöst wird je Sample  Σ_k q_k · EV_k(r) − Holdingkosten / r − Bridge = Preis × Aktien  per Bisektion.
EV_k(r) nutzt die gespeicherten FCFF-Pfade und Endwert-Bausteine der Engine (SegmentEVDetail). r ist eine
Unternehmens-IRR (implizite Kapitalkosten auf EV-Ebene, vor Hebel); die Eigenkapital-IRR folgt in Phase 4.
"""
from __future__ import annotations

import numpy as np

R_MAX = 1.0        # 100 % p. a.; darüber gilt die Rendite als „≥ R_MAX“
ITERATIONS = 60    # Bisektion: Intervall / 2^60 ⇒ weit unter 1 bp


def segment_ev_at(detail, r: np.ndarray) -> np.ndarray:
    """EV eines Segments bei Diskontsatz r (n,), gleiche Konvention wie compute_segment_ev."""
    years = np.arange(1, detail.forecast_years + 1, dtype=np.float64)
    exp = years - 0.5 if detail.mid_year else years
    pv_fcff = np.sum(detail.fcff / np.power((1.0 + r)[:, None], exp[None, :]), axis=1)
    if detail.tv_num is not None:
        tv = detail.tv_num / np.maximum(r - detail.tv_g, 1e-4)
    else:
        tv = detail.tv_abs
    return pv_fcff + tv / np.power(1.0 + r, float(detail.forecast_years))


def equity_at(results, r: np.ndarray) -> np.ndarray:
    """Eigenkapitalwert je Sample bei einheitlichem Diskontsatz r (Segmente und Holding)."""
    b = results.bridge_samples
    ev = sum(results.segment_ownership.get(k, 1.0) * segment_ev_at(d, r)
             for k, d in results.segment_cashflows.items())
    return (ev - b["holdingkosten"] / np.maximum(r, 1e-4) - b["nettoschulden"] - b["minderheiten"]
            - b["pensionen"] + b["nicht_operativ"] + b["beteiligungen"])


def implied_irr(results, price) -> np.ndarray:
    """IRR je Sample (n,) zum Preis je Aktie *price* (Skalar oder (n,)).

    Untergrenze ist je Sample das größte ewige Wachstum + 0,01 pp (sonst ist der Endwert nicht definiert).
    Liegt die Wurzel darunter (Preis über jedem endlichen Wert), wird die Untergrenze zurückgegeben; liegt sie
    über R_MAX, R_MAX. Beides markiert implied_irr_flags."""
    if not results.segment_cashflows:
        raise ValueError("Ergebnis ohne Cashflow-Bausteine (Engine vor v3 Phase 1?)")
    n = len(results.price_per_share)
    target = np.broadcast_to(np.asarray(price, float), (n,)) * results.bridge_samples["aktien"]
    gs = [d.tv_g for d in results.segment_cashflows.values() if d.tv_g is not None]
    lo = (np.max(np.column_stack(gs), axis=1) if gs else np.full(n, -0.5)) + 1e-4
    lo = np.maximum(lo, -0.5)
    hi = np.full(n, R_MAX)
    for _ in range(ITERATIONS):
        mid = (lo + hi) / 2
        above = equity_at(results, mid) > target       # Wert zu hoch ⇒ Rate muss steigen
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return (lo + hi) / 2


def irr_summary(results, price: float, rf: float) -> dict:
    """Kompakte IRR-Kennzahlen für das JSON-`ergebnis` (Anteile, nicht %)."""
    r = implied_irr(results, price)
    p5, p50, p95 = np.percentile(r, [5, 50, 95])
    return {"preis": float(price), "rf": float(rf), "p5": float(p5), "p50": float(p50), "p95": float(p95),
            "mittel": float(np.mean(r)), "vorsprung_p50": float(p50 - rf),
            "p_ueber_rf": float(np.mean(r > rf)), "anteil_am_rand": float(np.mean(r >= R_MAX - 1e-6))}


RAND_MAX_ANTEIL = 0.05   # mehr Samples am Rand ⇒ Bridge > Börsenwert, IRR nicht aussagekräftig (VW/MBG 11/2026)


def debt_like(results) -> float:
    """Median der Posten, die zum Fremdkapitalzins laufen (Nettoschulden, Pensionen; negativ = Nettokasse), für D/E.
    Minderheiten, Beteiligungen und nicht-operatives Vermögen wachsen mit dem Geschäft (Rendite ≈ r̄) und hebeln
    nicht: aus r̄·EV = r_E·E + k_d·D + r̄·M folgt r_E = r̄ + (r̄ − k_d)·D/E nur mit D ohne M."""
    b = results.bridge_samples
    return float(np.median(b["nettoschulden"] + b["pensionen"]))


def hold_returns(results, price: float, k_d: float, h: int) -> dict:
    """Rendite über *h* Jahre für „kaufen und halten“ (v3 Phase 4), je Sample (Zeilen bleiben gemeinsame Welten):
        R_i = (1 + r_E)^h · S_i − 1
    r̄ = Median der Unternehmens-IRR (Zins, zu dem der Kurs die mittlere Erwartung bezahlt), r_E = r̄ + D/E·(r̄ − k_d)
    die Eigenkapital-Rendite bei Hebel D/E zum Markt (k_d = Fremdkapitalkosten nach Steuer), S_i = Eigenkapitalwert
    des Szenarios i bei r̄ / Marktwert: die Überraschung in den Cashflows (Median 1, beschränkte Haftung ⇒ ≥ 0).
    Rendite kommt aus dem Kurs, Risiko aus den Cashflows – kein WACC-Urteil.
    # ponytail: Kursschwankung auf dem Weg fehlt bewusst (steckt größtenteils schon in S); Zwischenrisiko erst, wenn
    # Verkäufe vor h erzwungen werden können."""
    r = implied_irr(results, price)
    r_bar = float(np.median(r))
    aktien = results.bridge_samples["aktien"]
    d_e = debt_like(results) / (price * float(np.median(aktien)))
    r_e = max(r_bar + d_e * (r_bar - k_d), -0.99)
    S = np.maximum(equity_at(results, np.full(len(r), r_bar)), 0.0) / (price * aktien)
    return {"R": (1 + r_e) ** h * S - 1, "irr": r_bar, "irr_ek": r_e, "d_e": d_e,
            "irr_p5": float(np.percentile(r, 5)), "irr_p95": float(np.percentile(r, 95)),
            "am_rand": float(np.mean(r >= R_MAX - 1e-6))}
