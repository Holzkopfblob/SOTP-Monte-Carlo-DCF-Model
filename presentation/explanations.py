"""
Financial & statistical concept explanations (Markdown texts).

Each function returns a Markdown string to be embedded in
``st.expander`` blocks.  Keeping the prose here keeps the UI
helpers module lean and makes it easy to localise / update texts
without touching widget logic.
"""
from __future__ import annotations


def text_fcff() -> str:
    return r"""
**Free Cash Flow to Firm (FCFF)** ist der Cashflow, der allen Kapitalgebern
(Eigen- **und** Fremdkapital) nach Investitionen zur Verfügung steht:

$$FCFF = NOPAT + D\&A - CAPEX - \Delta NWC$$

| Komponente | Beschreibung | Typischer Bereich |
|---|---|---|
| **NOPAT** | Net Operating Profit After Tax = EBIT × (1 − Steuersatz) | – |
| **D&A** | Abschreibungen & Amortisation (nicht-zahlungswirksam) | 2–6 % vom Umsatz |
| **CAPEX** | Investitionsausgaben (neue Anlagen, Software etc.) | 3–15 % vom Umsatz |
| **ΔNWC** | Veränderung des Nettoumlaufvermögens | 5–20 % des Umsatzwachstums |

Der FCFF wird mit dem **WACC** diskontiert, um den **Enterprise Value** zu erhalten.
"""


def text_wacc() -> str:
    return r"""
**Weighted Average Cost of Capital (WACC)** ist der gewichtete
Kapitalkostensatz eines Unternehmens:

$$WACC = \frac{E}{V} \cdot k_e + \frac{D}{V} \cdot k_d \cdot (1 - t)$$

**CAPM:**  $k_e = r_f + \beta \cdot (r_m - r_f) + \text{Size Premium}$

Bei SOTP-Bewertungen sollte jedes Segment einen **eigenen WACC** erhalten,
da die Segmente unterschiedliche Risikoprofile haben.

> Ein um 1 Pp höherer WACC kann den Enterprise Value um **10–20 %** verändern.
"""


def text_distributions() -> str:
    return r"""
| Verteilung | Beschreibung | Typische Anwendung |
|---|---|---|
| **Fest** | Kein Zufall | Regulatorisch fixiert |
| **Normal** | Symmetrische Glockenkurve | Gleichmäßige Unsicherheit |
| **Log-Normal** | Rechtsschief, nur positiv | Wachstumsraten, Kurse |
| **Dreieck** | Min / Mode / Max | Schnelle Expertenschätzung |
| **Gleichverteilung** | Alle Werte gleich | Maximale Unsicherheit |
| **PERT** | Stärker zum Mode gewichtet | Professionelle Schätzung |

> **PERT > Dreieck** wenn der Mode gut begründet ist.
"""


def text_terminal_value() -> str:
    return r"""
Der **Terminal Value (TV)** erfasst den Unternehmenswert **nach** dem
Prognosezeitraum (typischerweise **60–80 %** des EV).

**Gordon Growth:**  $TV = \frac{FCFF_T \cdot (1+g)}{WACC - g}$
— Bedingung: $g < WACC$.

**Exit-Multiple:**  $TV = EBITDA_T \times Multiple$
— Intuitiver Marktbezug über Peer-Multiples.

> In der Praxis werden oft **beide** Methoden parallel gerechnet (Cross-Check).
"""


def text_monte_carlo() -> str:
    return r"""
**Monte-Carlo-Simulation** approximiert durch wiederholtes Ziehen von
Zufallsstichproben die Wahrscheinlichkeitsverteilung eines Ergebnisses.

| Klassisches DCF | Monte-Carlo DCF |
|---|---|
| Ein Szenario → ein Ergebnis | Tausende Szenarien → **Verteilung** |
| Unsicherheit ignoriert | Unsicherheit **quantifiziert** |

> Ab ~10.000 Iterationen sind Ergebnisse stabil; 50.000 liefern glattere Verteilungen.
"""


def text_corporate_bridge() -> str:
    return r"""
$$\text{Equity} = \sum EV_i - \frac{Holding}{r} - NetDebt - Minority - Pension + NonOp + Associates$$

| Posten | Erklärung |
|---|---|
| **Holdingkosten** | Zentrale Konzernkosten als Perpetuity |
| **Nettoverschuldung** | Finanzschulden − Cash |
| **Minderheitsanteile** | Drittanteile an Töchtern (−) |
| **Pensionen** | Unterdeckung Pensionsverpflichtungen (−) |
| **Nicht-op. Assets** | Überschuss-Cash, Immobilien (+) |
| **Beteiligungen** | At-Equity-Beteiligungen (+) |
"""


def text_interpretation() -> str:
    return """
- **Histogramm/KDE:** Häufigkeit → Breite = Unsicherheit
- **CDF:** P(Wert ≤ X) → Konfidenzaussagen
- **Tornado:** Spearman ρ → Feature Importance der Werttreiber
- **Waterfall:** Additive Zusammensetzung des Equity Values
- **P5/P95:** 90 %-Konfidenzintervall · **P25/P75:** Interquartilsbereich
"""


def text_sotp() -> str:
    return r"""
**Sum-of-the-Parts** bewertet jedes Segment individuell:

$$Equity = \sum EV_i - PV(Holding) - NetDebt$$

**Wann SOTP?** Konglomerate, heterogene Segmente, Spin-off-Analysen,
unterschiedliche Risikoprofile (→ verschiedene WACCs).
"""


def text_fade_model() -> str:
    return r"""
Das **Fade-Modell** lässt die Wachstumsrate exponentiell konvergieren:

$$g_t = g_T + (g_0 - g_T) \cdot e^{-\lambda t}$$

| Parameter | Bedeutung | Typische Werte |
|---|---|---|
| $g_0$ | Initiales Wachstum | 5–30 % |
| $g_T$ | Terminal-Wachstum | 1.5–3 % |
| $\lambda$ | Fade-Geschwindigkeit | 0.2 (langsam) – 1.5 (schnell) |

**Konstant** → reife Unternehmen · **Fade** → Wachstumsunternehmen
"""
