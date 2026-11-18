"""Baustein 5: fehlende Renditestreuung σ_x aus dem Rückblick-Labor schätzen.

Je Fall: e = tatsächliche Aktionärsrendite p. a. − Modell-Median (re_p50). Die Modellbänder (re_P5, re_P95) werden um
eine unabhängige Normalstreuung σ_x verbreitert: Halbbreite' = √(Halbbreite² + (1,645·σ_x)²). σ_x so, dass im
Lernzeitraum 90 % der (um die Median-Verschiebung bereinigten) Ergebnisse im Band liegen; einmal im Prüfzeitraum
kontrolliert. Horizonte 3 und 5 Jahre ⇒ Skalierung σ_x(h) = σ_x(5)·(5/h)^a.
Die Verschiebung (Modell zu optimistisch) wird nur ausgewiesen; sie gehört zu Baustein 3.

Aufruf:  python scripts/labor_kalibrierung.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.labor_auswertung import LERN, PRUEF, faelle_mit_ergebnis  # noqa: E402

Z = 1.645


def abdeckung(d, sigma, bias):
    e = d.tsr_pa_vorsichtig - d.re_p50 - bias
    lo = np.sqrt((d.re_p50 - d.re_p5) ** 2 + (Z * sigma) ** 2)
    hi = np.sqrt((d.re_p95 - d.re_p50) ** 2 + (Z * sigma) ** 2)
    return float(np.mean((e > -lo) & (e < hi)))


def sigma_fuer(d, bias, ziel=0.90):
    a, b = 0.0, 2.0
    for _ in range(50):
        m = (a + b) / 2
        a, b = (m, b) if abdeckung(d, m, bias) < ziel else (a, m)
    return (a + b) / 2


erg = {}
for h in (3, 5):
    d = faelle_mit_ergebnis(h)
    d = d[d.am_rand <= 0.05]
    lern, pruef = d[d.t.isin(LERN)], d[d.t.isin(PRUEF)]
    bias = float(np.median(lern.tsr_pa_vorsichtig - lern.re_p50))
    s = sigma_fuer(lern, bias)
    erg[h] = s
    print(f"h = {h} J.: Verschiebung (Median tatsächlich − Modell) {bias:+.1%} p. a. (Prüfen "
          f"{np.median(pruef.tsr_pa_vorsichtig - pruef.re_p50):+.1%}); Abdeckung ohne Korrektur {abdeckung(lern, 0, 0):.0%}"
          f" | σ_x = {s:.1%} p. a. ⇒ Lernen {abdeckung(lern, s, bias):.0%}, Prüfen {abdeckung(pruef, s, bias):.0%} "
          f"(Soll 90 %); Modell-Halbbreite Median {np.median((d.re_p95 - d.re_p5) / 2):.1%}")
a = np.log(erg[3] / erg[5]) / np.log(5 / 3)
print(f"Skalierung σ_x(h) = σ_x(5)·(5/h)^a mit a = {a:.2f} (0,5 = Zufallsweg, 1 = einmalige Neubewertung) "
      f"⇒ σ_x(10) = {erg[5] * (5 / 10) ** a:.1%} p. a.")
