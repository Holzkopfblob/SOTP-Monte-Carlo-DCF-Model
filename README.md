# 📊 SOTP Monte-Carlo DCF Modell + Portfolio-Optimierung

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-vektorisiert-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-287%20passed-brightgreen)](tests/)

> **Zwei unabhängige Streamlit-Apps** für professionelle Unternehmensbewertung und Portfoliostrukturierung — vollständig stochastisch, vektorisiert und interaktiv.

**App 1 – SOTP DCF**: Sum-of-the-Parts Unternehmensbewertung mit Monte-Carlo-Simulation über beliebig viele Geschäftssegmente. Fade-Modell für alle FCFF-Parameter, Cross-Segment-Korrelation via Gauss-Copula, Intra-Segment-Copula, 2 Sampling-Strategien (Pseudo-Random & Sobol), integrierte Bewertungsqualitäts-Metriken und erweiterte Insights (Tail Risk, Economic Profit, Conditional Sensitivity, Margin-of-Safety).

**App 2 – Portfolio-Optimierung**: Statistische Einzeltitel-Analyse, 8 Optimierungsmethoden (inkl. HRP, Black-Litterman, Min CVaR, Max Diversification, Multi-Asset Kelly), Ledoit-Wolf-Shrinkage, Efficient Frontier, clusterbasiertes Korrelationsmodell, 6 historische Krisenszenarien und Makro-Faktor-Sensitivitätsanalyse.

---

## 📑 Inhaltsverzeichnis

- [Highlights](#-highlights)
- [SOTP DCF App – Features](#-sotp-dcf-app--features)
- [Portfolio-Optimierung – Features](#-portfolio-optimierung--features)
- [Architektur](#-architektur)
- [Schnellstart](#-schnellstart)
- [Anwendungsbeispiel](#-anwendungsbeispiel)
- [Technologie-Stack](#-technologie-stack)
- [Mathematische Grundlagen](#-mathematische-grundlagen)
- [LLM-Research-Prompt](#-llm-research-prompt)
- [Projektstruktur](#-projektstruktur)
- [Tests](#-tests)
- [Mitwirken](#-mitwirken)

---

## ✨ Highlights

| Eigenschaft | Detail |
|---|---|
| **Vektorisierte MC-Engine** | 10.000–500.000 Iterationen via NumPy — keine Python-Loops über Simulationen |
| **2 Sampling-Strategien** | Pseudo-Random (Standard) und Quasi-MC Sobol (niedrige Diskrepanz) |
| **Phase 2 Insights** | Tail Risk (VaR/CVaR), Economic Profit, Conditional Sensitivity, Margin-of-Safety Dashboard, Percentile Convergence |
| **Universal-Fade-Modell** | Alle FCFF-Parameter konvergieren exponentiell zu Terminal-Werten: $p_t = p_T + (p_0 - p_T) \cdot e^{-\lambda t}$ |
| **Cross-Segment-Korrelation** | Gauss-Copula mit Cholesky-Dekomposition — stochastische Abhängigkeit zwischen Segmenten |
| **Intra-Segment-Copula** | 7×7-Gauss-Copula innerhalb jedes Segments — Abhängigkeiten zwischen Wachstum, Margen, WACC etc. |
| **Valuation Quality Score** | Composite-Metrik aus TV/EV-Ratio, Konvergenz, Sensitivity-Konzentration und Dispersions-Check |
| **Implied ROIC & Reinvestment** | Automatische Ableitung der implizierten Kapitalrendite aus den FCFF-Annahmen |
| **Stochastische Corporate Bridge** | Holdingkosten, Nettoverschuldung, Minderheiten, Pensionen u.v.m. optional als Verteilung |
| **6 Wahrscheinlichkeitsverteilungen** | Fest, Normal, Lognormal, Dreieck, Gleichverteilung, PERT — jeweils mit `ppf`-Methode für Copula-Sampling |
| **2 Terminal-Value-Methoden** | Gordon Growth Model & Exit-Multiple |
| **9 Portfolio-Optimierungen** | Max Sharpe, Min Vol, Risk Parity, Min CVaR, Max Diversification, Kelly, 1/N, **HRP**, **Black-Litterman** — ohne Radar-Chart (entfernt) |
| **Ledoit-Wolf Shrinkage** | Analytischer Kovarianz-Schätzer — robuster als Sample-Kovarianz bei wenigen Beobachtungen |
| **6 historische Krisenszenarien** | COVID-19, GFC 2008, Dot-Com, Euro-Krise, Inflationsschock, milde Korrektur |
| **Makro-Faktor-Sensitivität** | Sektor × {Zinsen, Inflation, BIP}-Sensitivitätsmatrix für 11 Branchen |
| **Clean Architecture** | 4-Schichten-Architektur — 51 Python-Dateien, ~9.200 LOC (6.500 App + 2.700 Tests) |
| **25+ interaktive Charts** | Histogramm+KDE, CDF, Tornado, Waterfall, Convergence, Fade-Preview, ROIC, Quality-Gauge, Margin-of-Safety, Economic Profit, Conditional Tornado, Percentile Convergence u.v.m. |
| **Excel-Export** | Summary, Assumptions, Raw-Data Sheets |
| **Save/Load** | JSON-Konfiguration speichern und laden |
| **287 Tests** | Vollständige Testsuite über alle Layer — 100 % Pass-Rate |

---

## 📊 SOTP DCF App – Features

### Bewertungsmodell

- **SOTP-Bewertung** – Bis zu 20 individuell konfigurierbare Geschäftssegmente
- **FCFF-Ansatz** – Free Cash Flow to Firm über 9 Werttreiber pro Segment:
  Revenue Growth, EBITDA-Marge, D&A, Steuersatz, CAPEX, NWC-Veränderung, WACC, Terminal Growth, Exit-Multiple
- **Universal-Fade-Modell** – **Alle** FCFF-Parameter konvergieren exponentiell von einem initialen zu einem Terminal-Wert. Konfigurierbarer λ-Fade-Speed mit interaktiver Vorschau
- **Cross-Segment-Korrelation** – Gauss-Copula mit frei konfigurierbarer n×n-Korrelationsmatrix über Segmente
- **Intra-Segment-Copula** – Optionale 7×7-Gauss-Copula innerhalb jedes Segments für korrelierte Werttreiber (z. B. Wachstum ↔ Marge)
- **Terminal Value** – Gordon Growth Model oder Exit-Multiple pro Segment, mit automatischen Guards (WACC-Floor, TV-Wachstum < WACC)
- **Equity Bridge** – Holdingkosten, Nettoverschuldung, Minderheitsanteile, Pensionsrückstellungen, nicht-operative Assets und Beteiligungen — jeweils optional als Verteilung

### Sampling-Strategien

| Methode | Beschreibung |
|---|---|
| **Pseudo-Random (Standard)** | Klassisches MC mit NumPy-RNG |
| **Quasi-MC (Sobol)** | Scrambled Sobol-Sequenz — niedrige Diskrepanz für schnellere Konvergenz |

> **Phase 1 Cleanup:** Antithetic Variates wurde entfernt — brachte in der Praxis wenig Varianzreduktion bei höherer Komplexität.

### Phase 2 Insights (Neue Features)

- **📊 Enriched Statistics** – Alle Statistik-Outputs jetzt mit Schiefe, Kurtosis, CV (Variationskoeffizient), IQR (Interquartilsabstand)
- **⚠️ Tail Risk** – Value-at-Risk (5%), Conditional VaR (Expected Shortfall), Tail Ratio für Downside-Quantifizierung
- **📈 Percentile Convergence** – Laufende P5/P50/P95-Stabilität zur Beurteilung, ob Tail-Perzentile konvergiert sind
- **🐻🐂 Conditional Sensitivity** – Tornado-Chart getrennt nach Bear (P<25%) vs. Bull (P>75%) Szenarien — identifiziert nicht-lineare Treiber
- **💰 Economic Profit (EVA)** – Segmentweiser EP = NOPAT - WACC × Invested Capital mit P(ROIC<WACC) als Value-Destruction-Wahrscheinlichkeit
- **🛡️ Margin-of-Safety Dashboard** – Interaktive Marktpreis-Eingabe → P(Upside), implizierte Rendite, Buy Price mit 30% MoS
- **📉 Normality Test** – Jarque-Bera + Shapiro-Wilk Tests zur Verteilungsklassifikation (Normal, Lognormal, Skew-Normal)

### Core Insights (Bewertungsqualität)

- **TV/EV-Dekomposition** – Segmentweise Analyse: wie viel des EV stammt aus dem Terminal Value (Warnsignal bei >70%)
- **Implied ROIC** – Automatisch abgeleitete Kapitalrendite aus FCFF-Annahmen als Plausibilitäts-Check
- **Reinvestment Rate** – Netto-Reinvestitionsquote zur Validierung der Wachstumsannahmen
- **Valuation Quality Score** – Composite-Metrik (0–100) aus TV/EV, Konvergenz, Sensitivity-Diversifikation, Dispersion

> **Phase 1 Cleanup:** SOTP Treemap entfernt — Waterfall-Chart deckt Segment-Dekomposition bereits ab.

### Stochastische Simulation

- **6 Verteilungstypen** – Jeder Parameter deterministisch oder stochastisch über 5 Verteilungen
- **Vektorisierte MC-Engine** – Komplett via NumPy, reproduzierbar über Seed
- **Sensitivity Analysis** – Spearman-Rangkorrelation identifiziert die einflussreichsten Werttreiber
- **Konvergenz-Diagnose** – Running Mean + 95 %-KI mit automatischer Bewertung

### Visualisierung & Export

- **25+ interaktive Charts** – EV/Equity-Histogramm+KDE, CDF, Tornado, SOTP-Waterfall, Konvergenz, Fade-Preview, TV/EV-Dekomposition, ROIC-Histogramm, ROIC-vs-WACC-Scatter, Quality-Gauge, Reinvestment-Rate, **Margin-of-Safety**, **Implied Return CDF**, **Economic Profit**, **Conditional Tornado**, **Percentile Convergence** u.v.m.
- **Statistik-Dashboard** – Mean, Median, Std, Schiefe, Kurtosis, CV, IQR, P5/P25/P75/P95, Min/Max
- **Excel-Export** – 3 Sheets (Summary, Assumptions, Raw Data)
- **JSON Save/Load** – Konfiguration exportieren/importieren

### 4-Tab-Oberfläche

| Tab | Inhalt |
|---|---|
| ⚙️ **Setup** | MC-Iterationen, Seed, **Sampling-Methode**, Segmentanzahl, Corporate Bridge, Cross-Segment-Korrelationsmatrix |
| 🏢 **Segmente** | Pro Segment: 9 stochastische Werttreiber, Fade-Terminals, TV-Methode, **Intra-Segment-Copula** |
| 🎲 **Simulation** | Konfigurationsübersicht, Start-Button |
| 📈 **Ergebnisse** | Metriken, Charts, Konvergenz, **Tail Risk**, TV/EV, Implied ROIC, Quality Score, **Margin-of-Safety**, **Economic Profit**, **Conditional Sensitivity**, Excel-Export |

---

## 💼 Portfolio-Optimierung – Features

### Einzeltitel-Analyse

- **Beliebig viele Aktien** – Fair-Value-Verteilung & aktueller Kurs pro Titel
- **6 Verteilungstypen** – Inkl. "Aus DCF-App (μ, σ, Schiefe)"-Modus
- **11 Kennzahlen pro Titel**: E[Rendite], P(Gewinn), Margin of Safety, Kelly f*, VaR (5%), CVaR, Sortino Ratio, Omega Ratio u.a.
- **Bewertungs-Ampel** – 🟢 Kaufen / 🟡 Halten / 🔴 Meiden

### Portfolio-Optimierung (8 Methoden)

| Methode | Beschreibung |
|---|---|
| **Gleichgewichtung (1/N)** | Naives Benchmark-Portfolio |
| **Max Sharpe Ratio** | Markowitz Mean-Variance, SLSQP-Optimierung |
| **Min Volatilität** | Minimales Portfoliorisiko |
| **Risk Parity** | Gleiche Risikobeiträge aller Assets |
| **Min CVaR** | Minimiert Expected Shortfall (Tail-Risk) |
| **Max Diversifikation** | Maximiert Diversification Ratio $DR = \frac{w'\sigma}{\sigma_p}$ |
| **Multi-Asset Kelly** | $\max w'\mu - \frac{1}{2}w'\Sigma w$ mit Half-Kelly |
| **HRP** *(Neu)* | Hierarchical Risk Parity — clusterbasiert, keine Matrixinversion nötig |

> **Black-Litterman** ist verfügbar, aber wird nicht in der Hauptvergleichstabelle angezeigt (nur wenn Views definiert sind).
> **Phase 1 Cleanup:** Radar-Chart entfernt — Gewichtungsvergleichs-Chart deckt Methodenvergleich visuell ab.

### Kovarianz-Schätzung

| Methode | Beschreibung |
|---|---|
| **Sample-Kovarianz** | Klassischer Schätzer |
| **Ledoit-Wolf (2004)** *(Neu)* | Analytische Shrinkage: $(1-\delta)S + \delta\mu I$ — robuster bei wenigen Beobachtungen, Shrinkage-Intensität automatisch optimal |

### Korrelation & Risiko

- **3 Korrelationsmodi** – Clusterbasiert (5 Sektorcluster), manuell, unkorreliert
- **PSD-Durchsetzung** – Eigenvalue-Clipping auf nächste positiv-semidefinite Matrix
- **Efficient Frontier** – 50-Punkt-Kurve + CML + 8 Portfolio-Punkte (+ Black-Litterman falls Views) + Einzelassets
- **Gewichtungs-Constraints** – Min/Max-Gewicht pro Asset
- **Diversifikationsanalyse** – Herfindahl-Index, effektive Anzahl Assets

### Stress-Tests

- **6 historische Krisenszenarien** *(Neu)*:

| Szenario | Marktschock | Korrelationsstress | Dauer |
|---|---|---|---|
| COVID-19 Crash (2020) | −35 % | 0.90 | 3 Mon. |
| GFC / Finanzkrise (2008) | −50 % | 0.95 | 18 Mon. |
| Dot-Com Crash (2001) | −40 % | 0.75 | 30 Mon. |
| Euro-Krise (2011) | −25 % | 0.85 | 9 Mon. |
| Inflation Shock (2022) | −20 % | 0.70 | 10 Mon. |
| Milde Korrektur | −15 % | 0.70 | 4 Mon. |

- **Sektorspezifische Schocks** – Jedes Szenario enthält individuelle Sektor-Schocks (z. B. GFC: Finanzen −40 %, Immobilien −35 %)
- **Makro-Faktor-Sensitivität** *(Neu)* – 11 Sektoren × 3 Faktoren (Zinsen, Inflation, BIP): wie verändert sich das Portfolio bei makroökonomischen Verschiebungen?
- **Manuelle Szenarien** – Marktschock-Slider, Korrelationsstress, gezielte Sektorkrisen
- **Normal vs. Stress** – Automatischer Vergleich mit Overlay-Charts

### 5-Tab-Oberfläche

| Tab | Inhalt |
|---|---|
| 📝 **Bewertungen** | Asset-Eingabe, **Kovarianz-Methode**, Korrelationsmatrix, **Black-Litterman Views**, JSON Save/Load |
| 🔍 **Einzeltitel** | Übersichtstabelle mit Signal & Omega Ratio, Detail-Analyse |
| 📊 **Portfolio** | 8 Methoden im Vergleich, Gewichte, Kennzahlen, Gewichtungsvergleichs-Chart, Renditeverteilung |
| 📈 **Efficient Frontier** | Frontier-Kurve, CML, 8 Portfolio-Punkte (+ Black-Litterman falls Views), Korrelations-Heatmap |
| ⚡ **Stress-Tests** | **6 historische Szenarien**, **Makro-Sensitivität**, manuelle Szenarien |

---

## 🏗 Architektur

Clean Architecture mit strikter 4-Schichten-Trennung und Page-Modul-Splitting:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Presentation Layer                                                  │
│  app.py · portfolio_app.py — Thin Orchestrators                      │
│  presentation/pages/ — 10 Page-Module (DCF: 4, Portfolio: 6)        │
│  presentation/charts.py · ui_helpers.py · explanations.py            │
├──────────────────────────────────────────────────────────────────────┤
│  Application Layer                                                   │
│  simulation_service.py · portfolio_service.py (Facade)               │
│  portfolio_analyser.py · portfolio_optimiser.py · portfolio_stress.py│
├──────────────────────────────────────────────────────────────────────┤
│  Domain Layer (reine Geschäftslogik, kein I/O)                       │
│  models.py · distributions.py · valuation.py · fade.py               │
│  valuation_metrics.py · statistics.py · portfolio_models.py          │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                                │
│  monte_carlo_engine.py · excel_export.py · config_io.py              │
└──────────────────────────────────────────────────────────────────────┘
```

**Kernprinzipien:**
- **Domain** enthält reine Datenstrukturen (`@dataclass`, `Enum`), Mathematik und Verteilungslogik — kein I/O
- **Infrastructure** kapselt MC-Engine (Copula, Sampling, Fade), Excel-Export, JSON-Serialisierung
- **Application** orchestriert Use Cases — `portfolio_service.py` als Facade über drei Spezial-Module
- **Presentation** ist austauschbar — Streamlit-Widgets, Plotly-Charts, UI-Erklärungen

---

## 🚀 Schnellstart

### Voraussetzungen

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Holzkopfblob/SOTP-Monte-Carlo-DCF-Model.git
cd SOTP-Monte-Carlo-DCF-Model
pip install -r requirements.txt
```

### Starten

```bash
# SOTP DCF Modell:
streamlit run app.py

# Portfolio-Optimierung (separater Port):
streamlit run portfolio_app.py --server.port 8502
```

---

## 📖 Anwendungsbeispiel

### SOTP DCF Workflow

1. **Setup** – Simulationsanzahl, Seed, **Sampling-Methode** (Sobol für schnelle Konvergenz), Segmentanzahl, Corporate Bridge
2. **Segmente konfigurieren** – Pro Segment: Verteilungen für alle 9 Werttreiber, optionale Fade-Terminal-Werte, Intra-Segment-Copula
3. **Simulation starten** – Ein Klick startet die vektorisierte MC-Engine
4. **Ergebnisse analysieren** – Histogramme, Tornado-Chart, SOTP-Waterfall, Implied ROIC, Quality Score, **Tail Risk**, **Margin-of-Safety**, **Economic Profit**, **Conditional Sensitivity**
5. **Export** – Excel-Report oder JSON-Konfiguration

### Portfolio Workflow

1. **Assets eingeben** – Fair-Value-Verteilung und Kurs pro Titel. **Kovarianz-Methode** wählen (Sample oder Ledoit-Wolf). Optional: **Black-Litterman Views** definieren
2. **Einzeltitel prüfen** – Ampelsystem + 11 Kennzahlen + 4 Charts
3. **Portfolio optimieren** – 8 Methoden im Vergleich (+ Black-Litterman falls Views), Gewichtungsvergleichs-Chart für visuellen Überblick
4. **Stress-Tests** – **6 historische Szenarien** + **Makro-Faktor-Analyse** + manuelle Szenarien

---

## 🔧 Technologie-Stack

| Komponente | Bibliothek | Zweck |
|---|---|---|
| Frontend | Streamlit ≥ 1.28 | Interaktive Web-Oberfläche |
| Numerik | NumPy ≥ 1.24 | Vektorisierte Berechnungen, RNG |
| Daten | Pandas ≥ 2.0 | Tabellarische Daten & Statistik |
| Statistik | SciPy ≥ 1.10 | Verteilungen, Optimierung (SLSQP), KDE, Quasi-MC (Sobol), Clustering |
| Visualisierung | Plotly ≥ 5.15 | 22+ interaktive Charts |
| Excel-Export | XlsxWriter ≥ 3.1 | Professionelle xlsx-Reports |

---

## 📐 Mathematische Grundlagen

### Free Cash Flow to Firm (FCFF)

$$FCFF = NOPAT + D\&A - CAPEX - \Delta NWC$$

wobei $NOPAT = (EBITDA - D\&A) \times (1 - t)$.

**Universal-Fade-Modell:** Jeder Parameter $p$ konvergiert exponentiell:
$$p_t = p_T + (p_0 - p_T) \cdot e^{-\lambda t}$$

### Terminal Value

**Gordon Growth:**
$$TV = \frac{FCFF_T \cdot (1+g)}{WACC - g}$$

**Exit-Multiple:**
$$TV = EBITDA_T \times Multiple$$

### DCF (Mid-Year Convention)

$$EV = \sum_{t=1}^{T} \frac{FCFF_t}{(1+WACC)^{t-0.5}} + \frac{TV}{(1+WACC)^T}$$

### Equity Value (SOTP Bridge)

$$Equity = \sum_i EV_i - PV(Holding) - NetDebt - Minority - Pensions + NonOp + Associates$$

### Cross-Segment-Korrelation (Gauss-Copula)

1. Cholesky: $L = \text{chol}(\Sigma)$
2. Korrelierte Normale: $Z = L \cdot Z_{\text{indep}}$
3. Uniforms: $U = \Phi(Z)$
4. Inversion: $X_i = F_i^{-1}(U_i)$

### Antithetic Variates

Für jeden Draw $u \sim U[0,1]$ wird gleichzeitig $1-u$ verwendet. Da $\text{Cov}(f(u), f(1-u)) < 0$ für monotone $f$, reduziert sich die Varianz des Mittelwert-Schätzers.

### Quasi-MC (Sobol)

Scrambled Sobol-Sequenz mit $2^m$ Samples — niedrige Diskrepanz sorgt für gleichmäßigere Abdeckung des Einheitshyperwürfels und schnellere Konvergenz ($O(1/n)$ statt $O(1/\sqrt{n})$).

### Ledoit-Wolf Shrinkage (2004)

$$\hat{\Sigma}_{LW} = (1-\delta)S + \delta \cdot \mu I$$

Analytisch optimale Shrinkage-Intensität $\delta \in [0,1]$ — Target ist skalierte Identität $\mu I$ mit $\mu = \text{tr}(S)/p$.

### Hierarchical Risk Parity (HRP)

1. Korrelationsbasierte Distanzmatrix: $d_{ij} = \sqrt{\frac{1}{2}(1-\rho_{ij})}$
2. Hierarchisches Clustering (Single-Linkage)
3. Quasi-Diagonalisierung über `leaves_list`
4. Rekursive Bisektionsallokation mit Inverse-Varianz-Gewichtung

### Black-Litterman

Posterior:
$$\mu_{BL} = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\pi + P'\Omega^{-1}q]$$

wobei $\pi = \delta \Sigma w_{eq}$ (Gleichgewichts-Prior), $P$ die View-Matrix und $\Omega$ die View-Unsicherheit darstellt.

### Multi-Asset Kelly Criterion

$$\max_w \; w'\mu - \frac{1}{2} w'\Sigma w \quad \text{s.t.} \; \sum w_i = 1,\; w_i \geq 0$$

Half-Kelly-Skalierung ($w/2$) für konservativere Positionsgrößen.

### Omega Ratio

$$\Omega = \frac{E[\max(R, 0)]}{E[\max(-R, 0)]}$$

### Bewertungsqualität

| Dimension | Gewicht | Gutes Ergebnis |
|---|---|---|
| TV/EV | 30 % | < 60 % |
| Konvergenz | 25 % | KI-Breite < 2 % |
| Sensitivity | 25 % | Kein Treiber dominiert |
| Dispersion | 20 % | CV < 30 % |

---

## 🤖 LLM-Research-Prompt

Im Verzeichnis `prompts/` liegt ein **strukturierter Research-Prompt** (`sotp_research_prompt.md`), der LLMs (GPT-4, Claude etc.) anleitet, alle benötigten Bewertungsparameter zu recherchieren:

- Segmentidentifikation aus Geschäftsberichten
- 10+ Parameter je Segment mit Verteilungsempfehlungen (inkl. Fade-Terminals)
- WACC via CAPM + Hamada-Unlever/Re-lever
- Erweiterte Corporate Bridge inkl. Minderheiten, Pensionen, nicht-operative Assets
- Cross-Segment- und Intra-Segment-Korrelation
- Sampling-Strategie-Empfehlung
- Plausibilitätschecks (Implied ROIC, TV/EV)

---

## 📁 Projektstruktur

```
SOTP-Monte-Carlo-DCF-Model/
│
├── app.py                              ← DCF App Entry Point (~138 LOC)
├── portfolio_app.py                    ← Portfolio App Entry Point (~95 LOC)
├── requirements.txt
├── pytest.ini
├── README.md
│
├── domain/                             ← Reine Geschäftslogik
│   ├── models.py                       ← Dataclasses, Enums, SamplingMethod (~241 LOC)
│   ├── distributions.py                ← 6 Verteilungsklassen + Factory + ppf (~154 LOC)
│   ├── valuation.py                    ← FCFF, TV, DCF (~203 LOC)
│   ├── fade.py                         ← Exponentieller Fade (~39 LOC)
│   ├── valuation_metrics.py            ← TV/EV, ROIC, Quality Score (~149 LOC)
│   ├── statistics.py                   ← Convergence-Metriken (~41 LOC)
│   └── portfolio_models.py             ← CovarianceMethod, InvestorView,
│                                         HistoricalScenario, Macro-Sensitivität (~200 LOC)
│
├── infrastructure/
│   ├── monte_carlo_engine.py           ← MC-Engine, Copula, Sampling, Fade (~516 LOC)
│   ├── excel_export.py                 ← xlsx-Report (~146 LOC)
│   └── config_io.py                    ← JSON Save/Load (~130 LOC)
│
├── application/
│   ├── simulation_service.py           ← Simulation + Sensitivity (~44 LOC)
│   ├── portfolio_service.py            ← Facade + Ledoit-Wolf (~303 LOC)
│   ├── portfolio_analyser.py           ← Einzeltitel-Analyse (~70 LOC)
│   ├── portfolio_optimiser.py          ← 9 Optimierungsmethoden inkl. HRP & BL (~420 LOC)
│   └── portfolio_stress.py             ← Stress + Hist. Szenarien + Makro (~205 LOC)
│
├── presentation/
│   ├── charts.py                       ← 22+ Plotly-Charts inkl. Radar & Treemap (~830 LOC)
│   ├── ui_helpers.py                   ← Streamlit-Widgets (~195 LOC)
│   ├── explanations.py                 ← UI-Erklärungstexte (~101 LOC)
│   └── pages/                          ← 10 Page-Module
│       ├── dcf_setup.py                ← Setup + Sampling-Methode (~214 LOC)
│       ├── dcf_segments.py             ← Segmente + Fade + Intra-Copula (~290 LOC)
│       ├── dcf_simulation.py           ← Simulations-Tab (~127 LOC)
│       ├── dcf_results.py              ← Ergebnisse + Treemap (~500 LOC)
│       ├── pf_common.py                ← Shared Utilities (~40 LOC)
│       ├── pf_input.py                 ← Eingabe + BL-Views + Cov-Methode (~382 LOC)
│       ├── pf_single.py                ← Einzeltitel-Analyse (~187 LOC)
│       ├── pf_portfolio.py             ← Portfolio + Radar-Chart (~186 LOC)
│       ├── pf_frontier.py              ← Efficient Frontier (~129 LOC)
│       └── pf_stress.py                ← Stress + Hist. Szenarien + Makro (~240 LOC)
│
├── prompts/
│   └── sotp_research_prompt.md         ← LLM-Research-Prompt
│
└── tests/                              ← 270 Tests (~2.685 LOC)
    ├── conftest.py                     ← Shared Fixtures
    ├── test_models.py                  ← Domain-Modelle
    ├── test_distributions.py           ← Verteilungsklassen + ppf
    ├── test_valuation.py               ← FCFF, TV, DCF
    ├── test_dist_mapping.py            ← Verteilungs-Factory
    ├── test_engine.py                  ← MC-Engine
    ├── test_config_io.py               ← JSON Save/Load
    ├── test_charts.py                  ← Chart-Rauchtest
    ├── test_ui_helpers.py              ← UI-Widgets
    ├── test_simulation_service.py      ← Simulation-Service
    ├── test_portfolio_models.py        ← Portfolio-Modelle
    ├── test_portfolio_service.py       ← Portfolio-Service
    ├── test_phase3.py                  ← Fade, Copula, ppf, Metriken
    └── test_phase4_to_8.py             ← Ledoit-Wolf, HRP, BL, Sampling,
                                          Szenarien, Makro, Radar, Treemap
```

**Gesamtumfang: ~9.200 Zeilen Code (6.500 App + 2.700 Tests) in 51 Python-Dateien**

---

## 🧪 Tests

**270 Tests** über alle Architektur-Schichten:

```bash
pytest -q
```

| Kategorie | Tests | Prüft |
|---|---|---|
| Domain | ~60 | Modelle, Verteilungen (inkl. ppf), FCFF, Fade, Metriken |
| Infrastructure | ~35 | MC-Engine, Copula, Sampling-Methoden, Excel, JSON |
| Application | ~55 | Simulation, Portfolio-Optimierung (9 Methoden), Ledoit-Wolf, BL, Stress |
| Presentation | ~20 | Chart-Rauchtest (inkl. Radar, Treemap), UI-Widgets |
| Integration | ~100 | Fade, Copula, ppf, Quality Score, TV/EV, ROIC, hist. Szenarien, Makro |

---

## 🤝 Mitwirken

1. **Fork** erstellen
2. **Feature-Branch**: `git checkout -b feature/mein-feature`
3. **Committen**: `git commit -m 'feat: Beschreibung'`
4. **Branch pushen**: `git push origin feature/mein-feature`
5. **Pull Request** öffnen

---

## 📄 Lizenz

[MIT-Lizenz](https://opensource.org/licenses/MIT)

---

<p align="center">
  Erstellt mit ❤️ und viel Statistik
</p>
