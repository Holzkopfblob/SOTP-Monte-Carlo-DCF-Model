"""
Reusable Streamlit UI Helper Functions.

Contains the distribution-input renderer and informational
explanation boxes for financial / statistical concepts.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from domain.models import DistributionConfig, DistributionType


# ═══════════════════════════════════════════════════════════════════════════
# Distribution type option labels (ordered)
# ═══════════════════════════════════════════════════════════════════════════

DIST_OPTIONS = [dt.value for dt in DistributionType]


# ═══════════════════════════════════════════════════════════════════════════
# Core input renderer
# ═══════════════════════════════════════════════════════════════════════════

def render_distribution_input(
    label: str,
    key_prefix: str,
    default_value: float = 0.0,
    is_percentage: bool = False,
    help_text: str = "",
) -> DistributionConfig:
    """
    Render Streamlit input widgets for one stochastic parameter.

    Parameters
    ----------
    label : str
        Human-readable parameter name.
    key_prefix : str
        Globally unique prefix for widget keys.
    default_value : float
        Default *display* value.  If ``is_percentage`` is True this is in
        % (e.g. 5 for 5 %).  Internally we store decimal fractions.
    is_percentage : bool
        If True, all user-facing values are interpreted as % and divided
        by 100 before storage.
    help_text : str
        Optional short explanation shown as caption.

    Returns
    -------
    DistributionConfig
        Fully populated config with decimal values.
    """
    pct_label = " (%)" if is_percentage else ""
    divisor = 100.0 if is_percentage else 1.0
    d = default_value  # shorthand for display default

    # ── Header row ────────────────────────────────────────────────────
    col_lbl, col_dist = st.columns([2, 3])
    with col_lbl:
        st.markdown(f"**{label}{pct_label}**")
        if help_text:
            st.caption(help_text)
    with col_dist:
        dist_type_str = st.selectbox(
            "Verteilung",
            options=DIST_OPTIONS,
            key=f"{key_prefix}_dtype",
            label_visibility="collapsed",
        )

    dist_type = DistributionType(dist_type_str)
    config = DistributionConfig(dist_type=dist_type)

    # ── Parameter inputs (conditional) ────────────────────────────────
    if dist_type == DistributionType.FIXED:
        val = st.number_input(
            f"Wert{pct_label}",
            value=d,
            key=f"{key_prefix}_fixed",
            format="%.4f",
        )
        config.fixed_value = val / divisor

    elif dist_type == DistributionType.NORMAL:
        c1, c2 = st.columns(2)
        mu = c1.number_input(
            "μ (Mittelwert)", value=d,
            key=f"{key_prefix}_n_mu", format="%.4f",
        )
        sigma = c2.number_input(
            "σ (Std.-Abw.)", value=max(abs(d) * 0.2, 0.01),
            key=f"{key_prefix}_n_sig", format="%.4f",
        )
        config.mean = mu / divisor
        config.std = sigma / divisor

    elif dist_type == DistributionType.LOGNORMAL:
        st.caption(
            "Geben Sie den **gewünschten** Mittelwert und die Std.-Abw. "
            "der Lognormalverteilung ein (nicht die der zugrunde liegenden "
            "Normalverteilung). Die Konvertierung erfolgt automatisch."
        )
        c1, c2 = st.columns(2)
        mu_disp = c1.number_input(
            "Gewünschter Mittelwert", value=max(d, 0.01),
            key=f"{key_prefix}_ln_mu", format="%.4f",
        )
        sig_disp = c2.number_input(
            "Gewünschte Std.-Abw.", value=max(abs(d) * 0.2, 0.01),
            key=f"{key_prefix}_ln_sig", format="%.4f",
        )
        desired_mean = mu_disp / divisor
        desired_std = sig_disp / divisor
        # Convert displayed mean/std → underlying normal μ, σ
        if desired_mean > 1e-12 and desired_std > 1e-12:
            sigma_sq = float(np.log(1.0 + (desired_std / desired_mean) ** 2))
            config.ln_sigma = float(np.sqrt(sigma_sq))
            config.ln_mu = float(np.log(desired_mean) - sigma_sq / 2.0)
        else:
            config.ln_mu = float(np.log(max(desired_mean, 1e-12)))
            config.ln_sigma = 0.1

    elif dist_type == DistributionType.TRIANGULAR:
        c1, c2, c3 = st.columns(3)
        lo = c1.number_input("Min", value=d * 0.8, key=f"{key_prefix}_tri_lo", format="%.4f")
        mo = c2.number_input("Mode", value=d, key=f"{key_prefix}_tri_mo", format="%.4f")
        hi = c3.number_input("Max", value=d * 1.2, key=f"{key_prefix}_tri_hi", format="%.4f")
        config.low = lo / divisor
        config.mode = mo / divisor
        config.high = hi / divisor

    elif dist_type == DistributionType.UNIFORM:
        c1, c2 = st.columns(2)
        lo = c1.number_input("Min", value=d * 0.8, key=f"{key_prefix}_uni_lo", format="%.4f")
        hi = c2.number_input("Max", value=d * 1.2, key=f"{key_prefix}_uni_hi", format="%.4f")
        config.low = lo / divisor
        config.high = hi / divisor

    elif dist_type == DistributionType.PERT:
        st.caption(
            "PERT-Verteilung: Mehr Gewicht auf dem wahrscheinlichsten Wert "
            "als die Dreiecksverteilung – ideal für Expertenschätzungen."
        )
        c1, c2, c3 = st.columns(3)
        lo = c1.number_input("Min", value=d * 0.7, key=f"{key_prefix}_pert_lo", format="%.4f")
        mo = c2.number_input("Mode", value=d, key=f"{key_prefix}_pert_mo", format="%.4f")
        hi = c3.number_input("Max", value=d * 1.3, key=f"{key_prefix}_pert_hi", format="%.4f")
        config.low = lo / divisor
        config.mode = mo / divisor
        config.high = hi / divisor

    return config


# ═══════════════════════════════════════════════════════════════════════════
# Concept explanation boxes
# ═══════════════════════════════════════════════════════════════════════════

def render_info_fcff() -> None:
    """Expandable info box explaining FCFF."""
    with st.expander("ℹ️ Was ist FCFF (Free Cash Flow to Firm)?"):
        st.markdown(r"""
**Free Cash Flow to Firm (FCFF)** ist der Cashflow, der allen Kapitalgebern
(Eigen- **und** Fremdkapital) nach Investitionen zur Verfügung steht:

$$FCFF = NOPAT + D\&A - CAPEX - \Delta NWC$$

| Komponente | Beschreibung | Typischer Bereich |
|---|---|---|
| **NOPAT** | Net Operating Profit After Tax = EBIT × (1 − Steuersatz) | – |
| **D&A** | Abschreibungen & Amortisation (nicht-zahlungswirksam) | 2–6 % vom Umsatz |
| **CAPEX** | Investitionsausgaben (neue Anlagen, Software etc.) | 3–15 % vom Umsatz |
| **ΔNWC** | Veränderung des Nettoumlaufvermögens (Forderungen + Vorräte − Verbindlichkeiten) | 5–20 % des Umsatzwachstums |

Der FCFF wird mit dem **WACC** diskontiert, um den **Enterprise Value** zu erhalten.

**Warum FCFF statt FCFE?**

Der FCFF-Ansatz bewertet das gesamte Unternehmen (Eigen- + Fremdkapital) und
ist unabhängig von der Kapitalstruktur.  Dies ist besonders bei SOTP-Bewertungen
vorteilhaft, da sich die Segmente ihre Finanzierungsstruktur nicht selbst wählen.
Das Fremdkapital wird erst in der **Unternehmensbrücke** (Net Debt) abgezogen.

**In diesem Modell:**
Wir vereinfachen, indem wir für jedes Segment das Umsatzwachstum als
konstante jährliche Rate modellieren und alle Kostenposten als Prozentsätze
vom Umsatz ausdrücken.  Die Monte-Carlo-Simulation variiert diese Raten
stochastisch, um die Unsicherheit abzubilden.
""")


def render_info_wacc() -> None:
    with st.expander("ℹ️ Was ist WACC? (ausführlich)"):
        st.markdown(r"""
**Weighted Average Cost of Capital (WACC)** ist der gewichtete
Kapitalkostensatz eines Unternehmens:

$$WACC = \frac{E}{V} \cdot k_e + \frac{D}{V} \cdot k_d \cdot (1 - t)$$

| Variable | Bedeutung | Typische Herleitung |
|---|---|---|
| $E/V$ | Eigenkapitalanteil | Marktkapitalisierung / (Marktkap. + Finanzschulden) |
| $k_e$ | Eigenkapitalkosten | CAPM: $r_f + \beta \cdot ERP$ |
| $D/V$ | Fremdkapitalanteil | Finanzschulden / (Marktkap. + Finanzschulden) |
| $k_d$ | Fremdkapitalkosten | Rendite bestehender Anleihen oder Zinsaufwand/Schulden |
| $t$ | Grenzsteuersatz | Effektiver Konzernsteuersatz |

**CAPM im Detail:**

$$k_e = r_f + \beta \cdot (r_m - r_f) + \text{Size Premium}$$

- $r_f$ = Risikofreier Zins (z.B. 10J-Bundesanleihe für EUR, 10J US Treasury für USD)
- $\beta$ = Sensitivität der Aktie gegenüber dem Markt (aus Regression oder Peer-Betas)
- $r_m - r_f$ = Equity Risk Premium (Damodaran: ~4.5–6.0 % je nach Markt)

**Segment-spezifische WACCs:**

Bei SOTP-Bewertungen sollte jedes Segment einen **eigenen WACC** erhalten,
da die Segmente unterschiedliche Risikoprofile haben. Dazu wird das Beta
von Pure-Play-Vergleichsunternehmen der jeweiligen Branche abgeleitet
(Unlever → Re-lever mit eigener Kapitalstruktur).

**Praxistipps:**
- Ein um 1 Prozentpunkt höherer WACC kann den Enterprise Value um **10–20 %** verändern.
- Bei Unsicherheit: PERT-Verteilung mit ±1–2 Prozentpunkten verwenden.
- WACC muss **immer > Terminal Growth** sein, sonst divergiert das Gordon Growth Model.
""")


def render_info_distributions() -> None:
    with st.expander("ℹ️ Wahrscheinlichkeitsverteilungen – Wann welche nutzen?"):
        st.markdown("""
### Übersicht

| Verteilung | Beschreibung | Typische Anwendung | Eingabe-Parameter |
|---|---|---|---|
| **Fest** | Kein Zufall – konstanter Wert | Regulatorisch fixiert (z.B. Steuersatz) | Wert |
| **Normal** | Symmetrische Glockenkurve | Gleichmäßige Unsicherheit um einen Mittelwert | μ, σ |
| **Log-Normal** | Rechtsschief, nur positive Werte | Wachstumsraten, Aktienkurse | Gewünschtes μ, σ |
| **Dreieck** | Min / Mode / Max definiert | Schnelle Expertenschätzung | Min, Mode, Max |
| **Gleichverteilung** | Alle Werte gleich wahrscheinlich | Maximale Unsicherheit in einem Intervall | Min, Max |
| **PERT** | Wie Dreieck, aber stärker zum Mode gewichtet | Professionelle Expertenschätzung | Min, Mode, Max |

### Welche Verteilung für welchen Parameter?

| Parameter | Empfohlene Verteilung | Begründung |
|---|---|---|
| **Umsatzwachstum** | PERT oder Normal | Experten können gut Min/Mode/Max schätzen; Normal wenn symmetrisch |
| **EBITDA-Marge** | PERT | Branchenkenntnis erlaubt realistische Dreipunktschätzung |
| **D&A (% Umsatz)** | Fest oder Dreieck | Relativ planbar aus historischen Daten |
| **Steuersatz** | Fest | Regulatorisch vorgegeben; Dreieck bei Steuerreform-Unsicherheit |
| **CAPEX (% Umsatz)** | PERT oder Dreieck | Management-Guidance oft als Range verfügbar |
| **NWC (% ΔUmsatz)** | Fest oder PERT | Branchenstabil; PERT bei zyklischen Geschäften |
| **WACC** | Normal oder PERT | **Höchste Sensitivität!** Mindestens ±1 Pp Unsicherheit einplanen |
| **TV-Wachstum** | PERT | Langfristiges BIP-Wachstum als Orientierung |
| **Exit-Multiple** | PERT oder Dreieck | Peer-Vergleich ergibt natürlichen Min/Mode/Max |

### PERT vs. Dreieck – Was ist der Unterschied?

- **Dreiecksverteilung:** Gleichmäßiger Übergang von Min zu Mode zu Max.
  Die Extremwerte haben relativ hohes Gewicht.
- **PERT-Verteilung:** Basiert auf einer Beta-Verteilung, die den **Mode stärker
  gewichtet** (Faktor λ=4). Extremwerte sind weniger wahrscheinlich.

> **Empfehlung:** PERT bevorzugen, wenn der wahrscheinlichste Wert (Mode) gut
> begründet ist. Dreieck nutzen, wenn alle drei Punkte gleich unsicher sind.
> Gleichverteilung nur bei **maximaler Unsicherheit** (keine Ahnung, wo der Wert liegt).
""")


def render_info_terminal_value() -> None:
    with st.expander("ℹ️ Terminal Value – Methoden (ausführlich)"):
        st.markdown(r"""
Der **Terminal Value (TV)** erfasst den Unternehmenswert **nach** dem
expliziten Prognosezeitraum.  Er macht typischerweise **60–80 %** des
gesamten Enterprise Values aus – die Wahl der Methode und Parameter
ist daher kritisch.

---

**Methode 1: Gordon Growth Model (Ewige Rente)**

$$TV = \frac{FCFF_{T} \cdot (1 + g)}{WACC - g}$$

| Parameter | Orientierungswerte |
|---|---|
| $g$ (ewiges Wachstum) | 1.5–3.0 % (≈ langfristiges nominales BIP-Wachstum) |
| Bedingung | $g < WACC$, sonst divergiert das Modell |
| Faustregel | $g < WACC - 3\text{Pp}$ für konservative Schätzung |

**Geeignet für:** Stabile Geschäftsmodelle mit vorhersehbaren Cashflows
(Versorger, Konsumgüter, reife Software).

**Vorsicht:** Da der TV = FCFF/(WACC−g), reagiert das Ergebnis **extrem sensitiv**
auf kleine Änderungen von g. Bei g=2.5% und WACC=8% ergibt sich ein
Faktor von 1/0.055 ≈ 18×. Steigt g auf 3%, ist der Faktor 1/0.05 = 20× (+11%).

---

**Methode 2: Exit-Multiple-Ansatz**

$$TV = EBITDA_{T} \times \text{EV/EBITDA-Multiple}$$

| Branche | Typische EV/EBITDA-Range |
|---|---|
| Software/SaaS | 15–30× |
| Industrie / Maschinenbau | 7–12× |
| Einzelhandel | 5–9× |
| Energie / Versorger | 6–10× |
| Pharma / Healthcare | 10–16× |

**Geeignet für:** Zyklische Branchen, M&A-aktive Sektoren und wenn
Peer-Group-Multiples gut verfügbar sind.

**Vorteil:** Intuitiverer Marktbezug. **Nachteil:** Impliziert eine
bestimmte Wachstumsrate, die nicht explizit modelliert wird.

---

> **Tipp:** In der Praxis werden oft **beide Methoden** parallel gerechnet
> und die Ergebnisse verglichen (Cross-Check).
""")


def render_info_monte_carlo() -> None:
    with st.expander("ℹ️ Was ist eine Monte-Carlo-Simulation?"):
        st.markdown(r"""
**Monte-Carlo-Simulation** ist ein statistisches Verfahren, das durch
wiederholtes Ziehen von Zufallsstichproben die Wahrscheinlichkeitsverteilung
eines Ergebnisses approximiert.

**Warum Monte Carlo statt einem einfachen DCF?**

| Klassisches DCF | Monte-Carlo DCF |
|---|---|
| Ein Szenario → ein Ergebnis | Tausende Szenarien → eine **Verteilung** |
| „Der Fair Value ist 42,50 €" | „Der Fair Value liegt mit 90 % zwischen 31 € und 58 €" |
| Unsicherheit wird ignoriert | Unsicherheit wird **quantifiziert** |

**Wie es funktioniert:**
1. Für jeden unsicheren Parameter (Wachstum, Marge, WACC etc.) wird eine
   Wahrscheinlichkeitsverteilung definiert.
2. Pro Iteration zieht der Algorithmus unabhängige Zufallswerte für alle Parameter.
3. Mit diesen Werten wird ein vollständiger DCF berechnet (FCFF → TV → EV → Equity).
4. Nach z.B. 50.000 Durchläufen entsteht eine Häufigkeitsverteilung, aus der
   **Erwartungswert, Konfidenzintervalle und Wahrscheinlichkeitsaussagen** ableitbar sind.

> **Faustregel:** Ab ca. 10.000 Iterationen sind die Ergebnisse in der Regel stabil.
> 50.000 liefern noch glattere Verteilungen bei moderater Rechenzeit.
""")


def render_info_corporate_bridge() -> None:
    with st.expander("ℹ️ Unternehmensbrücke – vom EV zum Equity Value"):
        st.markdown(r"""
Die **Unternehmensbrücke** übersetzt die Summe der Segment-Enterprise-Values
in den **Equity Value** (Eigenkapitalwert):

$$\text{Equity Value} = \underbrace{\sum_{i=1}^{n} EV_i}_{\text{Segment-EVs}}
- \underbrace{\frac{\text{Holdingkosten}}{r}}_{\text{PV Corp. Costs}}
- \underbrace{\text{Net Debt}}_{\text{Nettoverschuldung}}$$

| Komponente | Erklärung |
|---|---|
| **Holdingkosten** | Laufende Konzernkosten (Vorstand, zentrale IT, Compliance), die keinem Segment zugeordnet werden. Als Perpetuity diskontiert. |
| **Nettoverschuldung** | Finanzschulden − Cash. Ggf. plus Pensionsrückstellungen, Leasingverbindlichkeiten, Minderheitsanteile. |
| **Aktien ausstehend** | Voll verwässert: inkl. Optionen, RSUs, Wandelanleihen. Ergibt den **Preis je Aktie** = Equity Value / Shares. |
""")


def render_info_interpretation() -> None:
    with st.expander("ℹ️ Wie lese ich die Ergebnisse richtig?"):
        st.markdown("""
**Histogramm & KDE:** Zeigt die Häufigkeitsverteilung der simulierten Werte.
Je breiter die Verteilung, desto größer die Unsicherheit. Die Linie (KDE) glättet
das Histogramm zu einer stetigen Dichtefunktion.

**CDF (Kumulative Verteilungsfunktion):** Beantwortet die Frage:
*„Mit welcher Wahrscheinlichkeit ist der Wert ≤ X?"*
- Ablesen: Bei Y-Achse 0.20 und dem zugehörigen X-Wert heißt es:
  „Mit 20 % Wahrscheinlichkeit ist der Wert kleiner als X"
  → also mit **80 % größer** als X.

**Tornado-Chart (Sensitivität):** Zeigt per Spearman-Rangkorrelation, welcher
Input-Parameter den stärksten Einfluss auf den Equity Value hat.
- **Positive Korrelation** (grün): Steigt der Input, steigt der Equity Value
  (z.B. Umsatzwachstum, EBITDA-Marge).
- **Negative Korrelation** (rot): Steigt der Input, sinkt der Equity Value
  (z.B. WACC, Steuersatz, CAPEX).

**Waterfall-Chart:** Visualisiert die additive Zusammensetzung des Equity Values
aus den Segment-Erwartungswerten, abzüglich Holdingkosten und Net Debt.

**Konfidenzintervalle:**
- **P5 / P95** = 90 %-Konfidenzintervall: „In 9 von 10 Fällen liegt der Wert hier."
- **P25 / P75** = Interquartilsbereich (50 % der Szenarien).
""")


def render_info_sotp() -> None:
    with st.expander("ℹ️ Was ist Sum-of-the-Parts (SOTP)?"):
        st.markdown(r"""
**Sum-of-the-Parts** bewertet ein Unternehmen, indem jedes
Geschäftssegment **individuell** bewertet und anschließend aggregiert wird:

$$\text{Equity Value} = \sum_{i=1}^{n} EV_i - PV(\text{Holdingkosten}) - \text{Net Debt}$$

Dies ermöglicht eine differenziertere Bewertung als ein Single-Segment-DCF,
insbesondere bei Konglomeraten mit unterschiedlichen Wachstums- und
Risikoprofilen.

**Wann nutzt man SOTP?**
- Unternehmen mit **heterogenen Geschäftsbereichen** (z.B. Siemens: Automation + Healthcare)
- **Konglomerate**, bei denen ein einzelner WACC/Wachstumsrate nicht sinnvoll ist
- **Spin-off- oder Restrukturierungsanalysen** (Was wäre ein Segment alleine wert?)
- Wenn Segmente **unterschiedliche Risikoprofile** haben (→ verschiedene WACCs)

**Vorteil gegenüber Single-DCF:**
Jedes Segment erhält seinen eigenen Diskontierungssatz und Terminal-Value-Ansatz,
was die Bewertungsgenauigkeit erheblich verbessert.
""")


def render_info_fade_model() -> None:
    with st.expander("ℹ️ Revenue-Fade-Modell – Konvergierendes Wachstum"):
        st.markdown(r"""
### Warum ein Fade-Modell?

Das **konstante Wachstumsmodell** nimmt an, dass ein Segment über den gesamten
Prognosezeitraum mit der gleichen Rate wächst. Das ist **unrealistisch** für
wachstumsstarke Unternehmen, deren Wachstum sich mit zunehmender Reife abschwächt.

### Das Fade-Modell

Die Wachstumsrate konvergiert **exponentiell** vom initialen Wachstum $g_0$ zum
langfristigen Terminal-Wachstum $g_T$:

$$g_t = g_T + (g_0 - g_T) \cdot e^{-\lambda t}$$

| Parameter | Bedeutung | Typische Werte |
|---|---|---|
| $g_0$ | Initiales Umsatzwachstum (aktuell) | 5–30 % |
| $g_T$ | Terminal-Wachstum (langfristig) | 1.5–3 % |
| $\lambda$ | Fade-Geschwindigkeit | 0.2 (langsam) – 1.5 (schnell) |

### Wann welches Modell?

| Modell | Geeignet für |
|---|---|
| **Konstant** | Reife Unternehmen mit stabilem Wachstum (z.B. Versorger, Telekom) |
| **Fade** | Wachstumsunternehmen, deren Expansionsrate sich verlangsamt (z.B. SaaS, E-Commerce) |

> **Tipp:** Nutzen Sie den Vorschau-Chart unten, um den Wachstumspfad zu visualisieren,
> bevor Sie die Simulation starten.
""")
