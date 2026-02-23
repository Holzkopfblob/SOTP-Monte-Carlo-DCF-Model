# 📊 SOTP Monte-Carlo DCF Modell + Portfolio-Optimierung

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-vektorisiert-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-208%20passed-brightgreen)](tests/)

> **Zwei unabhängige Streamlit-Apps** für professionelle Unternehmensbewertung und Portfoliostrukturierung — vollständig stochastisch, vektorisiert und interaktiv.

**App 1 – SOTP DCF**: Sum-of-the-Parts Unternehmensbewertung mit Monte-Carlo-Simulation über beliebig viele Geschäftssegmente. Fade-Modell für alle FCFF-Parameter, Cross-Segment-Korrelation via Gauss-Copula und integrierte Bewertungsqualitäts-Metriken.

**App 2 – Portfolio-Optimierung**: Statistische Einzeltitel-Analyse, 7 Optimierungsmethoden (inkl. Min CVaR, Max Diversification, Multi-Asset Kelly), Efficient Frontier, clusterbasiertes Korrelationsmodell und Stress-Tests.

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
| **Universal-Fade-Modell** | Alle FCFF-Parameter (Wachstum, Margen, CAPEX, NWC …) konvergieren exponentiell zu Terminal-Werten: $p_t = p_T + (p_0 - p_T) \cdot e^{-\lambda t}$ |
| **Cross-Segment-Korrelation** | Gauss-Copula mit Cholesky-Dekompositon — stochastische Abhängigkeit zwischen Segmenten statt unabhängiger Draws |
| **Valuation Quality Score** | Composite-Metrik aus TV/EV-Ratio, Konvergenz, Sensitivity-Konzentration und Dispersions-Check |
| **Implied ROIC & Reinvestment** | Automatische Ableitung der implizierten Kapitalrendite und Reinvestitionsquote aus den FCFF-Annahmen |
| **Stochastische Corporate Bridge** | Holdingkosten, Nettoverschuldung, Aktienanzahl, Minderheiten, Pensionen u.v.m. optional als Verteilung |
| **6 Wahrscheinlichkeitsverteilungen** | Fest, Normal, Lognormal, Dreieck, Gleichverteilung, PERT — jeweils mit `ppf`-Methode für Copula-Sampling |
| **2 Terminal-Value-Methoden** | Gordon Growth Model & Exit-Multiple |
| **7 Portfolio-Optimierungen** | Max Sharpe, Min Volatilität, Risk Parity, Min CVaR, Max Diversification, Multi-Asset Kelly, Gleichgewichtung |
| **Clean Architecture** | 4-Schichten mit Page-Modul-Splitting — 47 Python-Dateien, 5.400+ LOC App + 2.100+ LOC Tests |
| **Interaktive Charts** | 20+ Plotly-Charts: Histogramm+KDE, CDF, Tornado, Waterfall, Convergence, Fade-Preview, TV/EV-Decomposition, ROIC, Quality-Gauge, Efficient Frontier u.v.m. |
| **Excel-Export** | Summary, Assumptions, Raw-Data für externe Audits |
| **Save/Load** | Gesamte Konfiguration als JSON speichern und laden |
| **208 Tests** | Vollständige Testsuite über alle Layer, 100 % grün |

---

## 📊 SOTP DCF App – Features

### Bewertungsmodell

- **SOTP-Bewertung** – Beliebige Anzahl an Geschäftssegmenten (bis 20), individuell konfigurierbar
- **FCFF-Ansatz** – Free Cash Flow to Firm über 9 Werttreiber pro Segment:
  - Revenue Growth, EBITDA-Marge, D&A, Steuersatz, CAPEX, NWC-Veränderung, WACC, Terminal Growth, Exit-Multiple
- **Universal-Fade-Modell** – Nicht nur Revenue Growth, sondern **alle FCFF-Parameter** können exponentiell von einem initialen zu einem Terminal-Wert konvergieren. Konfigurierbarer λ-Fade-Speed, interaktive Multi-Parameter-Vorschau
- **Cross-Segment-Korrelation** – Gauss-Copula mit frei konfigurierbarer n×n-Korrelationsmatrix. Automatische PSD-Validierung. Realistischere Modellierung als unabhängige Segments-Draws
- **Terminal Value** – Gordon Growth Model oder Exit-Multiple pro Segment, mit automatischen Guards (WACC-Floor, TV-Wachstum < WACC)
- **Erweiterte Equity Bridge** – Holdingkosten, Nettoverschuldung, Aktienanzahl, Minderheitsanteile, Pensionsrückstellungen, nicht-operative Assets und Beteiligungen — jeweils optional als Verteilung modellierbar

### Core Insights (Bewertungsqualität)

- **TV/EV-Dekomposition** – Segmentweise Aufschlüsselung: wie viel des Enterprise Value aus dem Terminal Value stammt (Nachhaltigkeitscheck)
- **Implied ROIC** – Automatisch abgeleitete Kapitalrendite aus den FCFF-Annahmen — inkonsistente Parameterkombinationen werden sichtbar
- **Reinvestment Rate** – Netto-Reinvestitionsquote pro Segment als Plausibilitätsprüfung
- **Valuation Quality Score** – Composite-Metrik (0–100) aus vier Dimensionen: TV/EV, Konvergenz, Sensitivity-Konzentration, Dispersions-Check

### Stochastische Simulation

- **6 Verteilungstypen** – Jeder der 9 Parameter kann deterministisch (fest) oder stochastisch über 5 Verteilungen modelliert werden
- **Vektorisierte MC-Engine** – Komplett via NumPy (keine For-Schleifen), reproduzierbar über Seed
- **Sensitivity Analysis** – Spearman-Rangkorrelation identifiziert automatisch die einflussreichsten Werttreiber (inkl. stochastischer Bridge-Parameter)
- **Konvergenz-Diagnose** – Running Mean + 95 %-Konfidenzintervall zeigt, ob die Simulationsanzahl ausreicht. Automatische Bewertung: ✅ Konvergiert / ⚠️ Akzeptabel / ❌ Nicht konvergiert

### Visualisierung & Export

- **20+ interaktive Charts** – EV-Histogramm+KDE, Equity-Histogramm, CDF mit Perzentilen, Preis/Aktie-Histogramm, Tornado-Chart, SOTP-Waterfall, Konvergenz-Chart, Revenue-Fade-Preview, Parameter-Fade-Preview, TV/EV-Dekomposition, Implied-ROIC, Quality-Score-Gauge, Quality-Breakdown u.v.m.
- **Statistik-Dashboard** – Mean, Median, Std, P5/P25/P75/P95, Min/Max für EV, Equity und Preis/Aktie
- **Excel-Export** – 3 Sheets (Summary & Statistics, Segment Assumptions, Raw Simulation Data bis 100k Zeilen)
- **JSON Save/Load** – Komplette Konfiguration exportieren/importieren
- **Portfolio-App-Link** – μ, σ, Schiefe, Kurtosis und Perzentile als Kopiervorlage für die Portfolio-App

### 4-Tab-Oberfläche

| Tab | Inhalt |
|---|---|
| ⚙️ **Setup** | MC-Iterationen, Seed, Segmentanzahl, Corporate Bridge (optional stochastisch), Cross-Segment-Korrelationsmatrix |
| 🏢 **Segmente** | Pro Segment: Name, Basisumsatz, Prognosejahre, Wachstumsmodell (Konstant/Fade), 9 stochastische Werttreiber, optionale Terminal-Zielwerte für alle Parameter, TV-Methode |
| 🎲 **Simulation** | Konfigurationsübersicht, Start-Button mit Fortschrittsbalken |
| 📈 **Ergebnisse** | Metriken, Charts, Konvergenz-Diagnose, TV/EV-Dekomposition, Implied ROIC, Quality Score Dashboard, Segment-Details, Excel-Export |

---

## 💼 Portfolio-Optimierung – Features

### Einzeltitel-Analyse

- **Beliebig viele Aktien** – Fair-Value-Verteilung & aktueller Kurs pro Titel
- **6 Verteilungstypen** – Inkl. speziellem "Aus DCF-App (μ, σ, Schiefe)"-Modus für nahtlose Integration
- **11 Kennzahlen pro Titel**:
  - E[Rendite], P(Gewinn), Margin of Safety, Kelly f* (inkl. Half-Kelly)
  - VaR (5%), CVaR / Expected Shortfall, Sortino Ratio, **Omega Ratio**
- **Bewertungs-Ampel** – 🟢 Kaufen / 🟡 Halten / 🔴 Meiden
- **4 Charts pro Titel** – FV-Histogramm, CDF, Renditeverteilung, Up-/Downside-Bars

### Portfolio-Optimierung

| Methode | Beschreibung |
|---|---|
| **Max Sharpe Ratio** | Markowitz Mean-Variance, SLSQP-Optimierung |
| **Min Volatilität** | Minimales Portfoliorisiko bei gegebenen Assets |
| **Risk Parity** | Gleiche Risikobeiträge aller Assets |
| **Min CVaR** | Minimiert Expected Shortfall (Tail-Risk) aus MC-Samples |
| **Max Diversification** | Maximiert Diversification Ratio $DR = \frac{w'\sigma}{\sigma_p}$ |
| **Multi-Asset Kelly** | Vollständiges Kelly-Kriterium $\max w'\mu - \frac{1}{2}w'\Sigma w$ mit Half-Kelly-Skalierung |
| **Gleichgewichtung (1/N)** | Naives Benchmark-Portfolio |

### Korrelation & Risiko

- **3 Korrelationsmodi** – Clusterbasierte Sektorkorrelation (5 ökonomische Cluster: Growth, Cyclical, Defensive, Financial, Energy), manuell, oder unkorreliert
- **PSD-Durchsetzung** – Automatische Projektion auf die nächste positiv-semidefinite Matrix via Eigenvalue-Clipping
- **Efficient Frontier** – 50-Punkt-Kurve + Capital Market Line + 7 Portfolio-Punkte + Einzelassets
- **Gewichtungs-Constraints** – Min/Max-Gewicht pro Asset
- **Diversifikationsanalyse** – Herfindahl-Index, effektive Anzahl Assets

### Stress-Tests

- **3 Stress-Presets** – COVID-19 Crash, GFC 2008, Mild Correction (vorkonfiguriert)
- **Marktschock** – Slider für universellen Kursrückgang
- **Korrelations-Stress** – Erhöhung aller Korrelationen gegen 1.0
- **Sektorkrisen** – Gezielte Schocks auf einzelne Sektoren
- **Normal vs. Stress** – Automatischer Vergleich mit Interpretation und Overlay-Charts

### 5-Tab-Oberfläche

| Tab | Inhalt |
|---|---|
| 📝 **Bewertungen** | Asset-Eingabe, Korrelationsmatrix, Constraints, JSON Save/Load |
| 🔍 **Einzeltitel** | Übersichtstabelle mit Signal & Omega Ratio, Detail-Analyse pro Aktie |
| 📊 **Portfolio** | 7 Optimierungsmethoden im Vergleich, Gewichtungs-Chart, Renditeverteilung |
| 📈 **Efficient Frontier** | Frontier-Kurve, CML, 7 Portfolio-Punkte, Korrelations-Heatmap |
| ⚡ **Stress-Tests** | 3 Stress-Presets + manuelle Szenarien mit Overlay-Charts |

---

## 🏗 Architektur

Das Projekt folgt **Clean Architecture** Prinzipien mit strikter Schichtentrennung und Page-Modul-Splitting:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Presentation Layer                                                 │
│  app.py (~130 LOC) · portfolio_app.py (~95 LOC) — Thin Orchestrators│
│  presentation/pages/ — 11 Page-Module (DCF: 4, Portfolio: 6)       │
│  presentation/charts.py · presentation/ui_helpers.py               │
├─────────────────────────────────────────────────────────────────────┤
│  Application Layer                                                  │
│  simulation_service.py · portfolio_service.py (Facade)             │
│  portfolio_analyser.py · portfolio_optimiser.py · portfolio_stress.py│
├─────────────────────────────────────────────────────────────────────┤
│  Domain Layer (reine Geschäftslogik, kein I/O)                      │
│  models.py · distributions.py · valuation.py · fade.py             │
│  valuation_metrics.py · statistics.py · portfolio_models.py        │
├─────────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                               │
│  monte_carlo_engine.py · excel_export.py · config_io.py            │
└─────────────────────────────────────────────────────────────────────┘
```

**Kernprinzipien:**

- **Domain Layer** enthält reine Datenstrukturen (`@dataclass`, `Enum`), Mathematik und Verteilungslogik — kein Framework-Code, kein I/O
- **Infrastructure** kapselt die MC-Engine (inkl. Copula-Sampling, Parameter-Fade), Excel-Export und JSON-Serialisierung
- **Application** orchestriert Use Cases (Simulation, Sensitivity, Portfolio-Optimierung) – `portfolio_service.py` als dünne Facade über drei Spezial-Module
- **Presentation** ist austauschbar — Streamlit-Widgets, Plotly-Charts, UI-Erklärungen. Monolithische Apps wurden in Page-Module aufgeteilt

---

## 🚀 Schnellstart

### Voraussetzungen

- Python 3.10+
- pip

### Installation

```bash
# Repository klonen
git clone https://github.com/Holzkopfblob/SOTP-Monte-Carlo-DCF-Model.git
cd SOTP-Monte-Carlo-DCF-Model

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Starten

```bash
# SOTP DCF Modell:
streamlit run app.py

# Portfolio-Optimierung (separater Port):
streamlit run portfolio_app.py --server.port 8502
```

Beide Apps öffnen sich automatisch im Browser unter `http://localhost:8501` bzw. `http://localhost:8502`.

---

## 📖 Anwendungsbeispiel

### SOTP DCF Workflow

1. **Setup** – Wähle 10.000–500.000 Simulationen, setze einen Seed für Reproduzierbarkeit, definiere die Anzahl der Geschäftssegmente und die Corporate Bridge (Holdingkosten, Net Debt, Aktienanzahl)
2. **Segmente konfigurieren** – Für jedes Segment: Name, Basisumsatz, Prognosejahre (1–10), dann je Parameter eine Verteilung wählen (z. B. Revenue Growth als PERT mit 3%/7%/12%)
3. **Simulation starten** – Ein Klick startet die vektorisierte MC-Engine
4. **Ergebnisse analysieren** – Histogramme zeigen die Verteilung von Enterprise Value, Equity Value und Preis/Aktie. Der Tornado-Chart zeigt, welche Parameter den größten Einfluss haben. Der Waterfall visualisiert die SOTP-Wertbrücke
5. **Export** – Excel-Report herunterladen oder Konfiguration als JSON speichern

### Portfolio-Optimierung Workflow

1. **Assets eingeben** – Für jede Aktie: Fair-Value-Schätzung (Verteilung) und aktuellen Kurs. Optional: Clusterbasierte Korrelationsmatrix und Gewichtungs-Constraints definieren
2. **Einzeltitel prüfen** – Ampelsystem zeigt auf einen Blick: Kaufen, Halten oder Meiden. Detailansicht mit 11 Kennzahlen (inkl. Omega Ratio) + 4 Charts pro Titel
3. **Portfolio optimieren** – 7 Methoden im direkten Vergleich. Efficient Frontier zeigt das Risiko-Rendite-Spektrum
4. **Stress-Tests** – Vorkonfigurierte Presets (COVID-19, GFC 2008) oder manuelle Szenarien zur Robustheitsprüfung

---

## 🔧 Technologie-Stack

| Komponente | Bibliothek | Zweck |
|---|---|---|
| Frontend/GUI | Streamlit ≥ 1.28 | Interaktive Web-Oberfläche |
| Numerik | NumPy ≥ 1.24 | Vektorisierte Berechnungen, RNG |
| Datenverarbeitung | Pandas ≥ 2.0 | Tabellarische Daten & Statistik |
| Statistik | SciPy ≥ 1.10 | Verteilungen, Optimierung (SLSQP), KDE |
| Visualisierung | Plotly ≥ 5.15 | Interaktive Charts |
| Excel-Export | XlsxWriter ≥ 3.1 | Professionelle xlsx-Reports |
| Plotting (optional) | Matplotlib ≥ 3.7 | Ergänzende Visualisierungen |

---

## 📐 Mathematische Grundlagen

### Free Cash Flow to Firm (FCFF)

$$FCFF = NOPAT + D\&A - CAPEX - \Delta NWC$$

wobei $NOPAT = (EBITDA - D\&A) \times (1 - t)$.

**Konstantes Wachstum:** $R_t = R_0 \cdot (1+g)^t$

**Universal-Fade-Modell:** Jeder Parameter $p$ konvergiert exponentiell von seinem initialen zum terminalen Wert:
$$p_t = p_T + (p_0 - p_T) \cdot e^{-\lambda t}$$

Dies gilt für Revenue Growth, EBITDA-Marge, D&A%, Steuersatz, CAPEX% und NWC% — jeweils unabhängig konfigurierbar. Der Umsatz wird iterativ berechnet: $R_t = R_{t-1} \cdot (1 + g_t)$, während die übrigen Parameter direkt zeitvariable (n, T)-Matrizen erzeugen.

### Terminal Value

**Gordon Growth Model:**
$$TV = \frac{FCFF_T \cdot (1+g)}{WACC - g}$$

**Exit-Multiple:**
$$TV = EBITDA_T \times Multiple$$

### Enterprise Value (DCF)

Mid-Year Convention (Standard):
$$EV = \sum_{t=1}^{T} \frac{FCFF_t}{(1+WACC)^{t-0.5}} + \frac{TV}{(1+WACC)^T}$$

### Equity Value (SOTP Bridge)

$$Equity = \sum_i EV_i - PV(Holdingkosten) - Net\;Debt - Minority - Pensions + Non\text{-}Op\;Assets + Associates$$

Dabei können alle Bridge-Parameter jeweils **deterministisch oder stochastisch** modelliert werden.

### Cross-Segment-Korrelation (Gauss-Copula)

1. Cholesky-Dekompositon der Korrelationsmatrix $\Sigma$: $L = \text{chol}(\Sigma)$
2. Gemeinsame Standardnormalvariablen: $Z = L \cdot Z_{\text{indep}}$ mit $Z_{\text{indep}} \sim N(0, I)$
3. Transformation in korrelierte Uniforms: $U = \Phi(Z)$
4. Inversion über die jeweilige Verteilung: $X_i = F_i^{-1}(U_i)$

So werden parameterübergreifende Abhängigkeiten zwischen Segmenten modelliert, ohne die Marginalverteilungen zu verändern.

### Bewertungsqualität (Core Insights)

**TV/EV-Ratio:** $\rho_{TV} = PV(TV) / EV$ — misst die Nachhaltigkeit der Bewertung (< 60 % = gut)

**Implied ROIC:**
$$ROIC = \frac{NOPAT\;Margin}{Reinvestment\;Rate} \cdot g$$

**Quality Score:** Composite-Metrik (0–100) aus:
| Dimension | Gewicht | Gutes Ergebnis |
|---|---|---|
| TV/EV | 30 % | < 60 % |
| Konvergenz | 25 % | KI-Breite < 2 % |
| Sensitivity | 25 % | Kein Treiber dominiert |
| Dispersion | 20 % | CV < 30 % |

### Multi-Asset Kelly Criterion

**Multi-Asset (Portfolio):**
$$\max_w \; w'\mu - \frac{1}{2} w'\Sigma w \quad \text{s.t.} \; \sum w_i = 1,\; w_i \geq 0$$

gekappt auf $[0, 1]$ — die App verwendet Half-Kelly ($w/2$) zur konservativeren Positionsgrößenbestimmung.

### Omega Ratio

$$\Omega = \frac{E[\max(R, 0)]}{E[\max(-R, 0)]}$$

Verhältnis der erwarteten Gewinne zu den erwarteten Verlusten — $\Omega > 1$ signalisiert eine asymmetrisch positive Renditeverteilung.

### PERT-Verteilung

Re-skalierte Beta-Verteilung mit $\lambda = 4$:

$$\alpha_1 = 1 + \lambda \cdot \frac{mode - min}{max - min}, \quad \alpha_2 = 1 + \lambda \cdot \frac{max - mode}{max - min}$$

---

## 🤖 LLM-Research-Prompt

Im Verzeichnis `prompts/` liegt ein **strukturierter Research-Prompt** (`sotp_research_prompt.md`), der LLMs (GPT-4, Claude, etc.) anleitet, alle benötigten Bewertungsparameter für ein Unternehmen zu recherchieren:

- Segmentidentifikation aus Geschäftsberichten
- **10+ Parameter je Segment** mit Verteilungsempfehlungen (inkl. Terminal-Zielwerte für Fade-Modell)
- WACC-Ableitung via CAPM + Hamada-Unlever/Re-lever
- **Erweiterte Corporate Bridge** — Holdingkosten, Net Debt, Aktienanzahl, Minderheitsanteile, Pensionsrückstellungen, nicht-operative Assets, Beteiligungen
- **Cross-Segment-Korrelation** — Anleitung zur Schätzung paarweiser Abhängigkeiten zwischen Segmenten
- Plausibilitätschecks (inkl. Implied ROIC & TV/EV-Ratio) und Quellenverzeichnis

Das Ausgabeformat ist direkt in die Streamlit-App übertragbar.

---

## 📁 Projektstruktur

```
SOTP-Monte-Carlo-DCF-Model/
│
├── app.py                              ← Streamlit Entry Point – SOTP DCF (~130 LOC)
├── portfolio_app.py                    ← Streamlit Entry Point – Portfolio (~95 LOC)
├── requirements.txt                    ← Python-Abhängigkeiten
├── pytest.ini                          ← Pytest-Konfiguration
├── README.md                           ← Diese Datei
│
├── domain/                             ← Reine Geschäftslogik (kein I/O)
│   ├── models.py                       ← Dataclasses, Enums, RevenueGrowthMode (~185 LOC)
│   ├── distributions.py                ← 6 Verteilungsklassen + Factory + ppf (~155 LOC)
│   ├── valuation.py                    ← FCFF (Fade + Konstant), TV, DCF (~200 LOC)
│   ├── fade.py                         ← build_fade_curve() – exponentieller Fade (~40 LOC)
│   ├── valuation_metrics.py            ← TV/EV, Implied ROIC, Quality Score (~150 LOC)
│   ├── statistics.py                   ← Convergence-Metriken (~20 LOC)
│   └── portfolio_models.py             ← Portfolio-Dataclasses (~85 LOC)
│
├── infrastructure/                     ← Technische Infrastruktur
│   ├── monte_carlo_engine.py           ← MC-Engine, Copula, Parameter-Fade (~405 LOC)
│   ├── excel_export.py                 ← xlsx-Report mit 3 Sheets (~145 LOC)
│   └── config_io.py                    ← JSON Save/Load (~100 LOC)
│
├── application/                        ← Use-Case-Orchestrierung
│   ├── simulation_service.py           ← Simulation + Sensitivity (~55 LOC)
│   ├── portfolio_service.py            ← Portfolio-Facade (~250 LOC)
│   ├── portfolio_analyser.py           ← Einzeltitel-Analyse (~70 LOC)
│   ├── portfolio_optimiser.py          ← 7 Optimierungsmethoden (~290 LOC)
│   └── portfolio_stress.py             ← Stress-Test-Engine (~105 LOC)
│
├── presentation/                       ← UI & Visualisierung
│   ├── charts.py                       ← 20+ Plotly-Charts (~645 LOC)
│   ├── ui_helpers.py                   ← Streamlit-Widgets (~195 LOC)
│   ├── explanations.py                 ← UI-Erklärungstexte (~100 LOC)
│   └── pages/                          ← 11 Page-Module
│       ├── dcf_setup.py                ← Setup-Tab inkl. Korrelationsmatrix (~260 LOC)
│       ├── dcf_segments.py             ← Segmente-Tab inkl. Fade-Terminals (~250 LOC)
│       ├── dcf_simulation.py           ← Simulations-Tab (~100 LOC)
│       ├── dcf_results.py              ← Ergebnisse-Tab inkl. Quality Score (~435 LOC)
│       ├── pf_common.py                ← Portfolio-Shared Utilities (~40 LOC)
│       ├── pf_input.py                 ← Portfolio-Eingabe (~295 LOC)
│       ├── pf_single.py                ← Einzeltitel-Analyse (~190 LOC)
│       ├── pf_portfolio.py             ← Portfolio-Optimierung (~140 LOC)
│       ├── pf_frontier.py              ← Efficient Frontier (~125 LOC)
│       └── pf_stress.py                ← Stress-Tests (~135 LOC)
│
├── prompts/
│   └── sotp_research_prompt.md         ← LLM-Research-Prompt
│
└── tests/                              ← 208 Tests (~2.080 LOC)
    ├── conftest.py                     ← Shared Fixtures
    ├── test_models.py                  ← Domain-Modelle
    ├── test_distributions.py           ← Verteilungsklassen + ppf
    ├── test_valuation.py               ← FCFF, TV, DCF-Berechnung
    ├── test_dist_mapping.py            ← Verteilungs-Factory
    ├── test_engine.py                  ← MC-Engine
    ├── test_config_io.py               ← JSON Save/Load
    ├── test_charts.py                  ← Chart-Rauchtest
    ├── test_ui_helpers.py              ← UI-Widgets
    ├── test_simulation_service.py      ← Simulation-Service
    ├── test_portfolio_models.py        ← Portfolio-Modelle
    ├── test_portfolio_service.py       ← Portfolio-Service
    └── test_phase3.py                  ← Phase-3: Fade, Copula, ppf, Metriken
```

**Gesamtumfang: ~7.500 Zeilen Code (5.400 App + 2.100 Tests) in 47 Python-Dateien**

---

## 🧪 Tests

**208 Tests** über alle Architektur-Schichten, ausführbar via:

```bash
pytest -q
```

| Kategorie | Tests | Prüft |
|---|---|---|
| Domain | ~60 | Modelle, Verteilungen (inkl. ppf), FCFF-Berechnung, Fade-Kurven, Metriken |
| Infrastructure | ~35 | MC-Engine, Copula-Sampling, Excel-Export, JSON-I/O |
| Application | ~25 | Simulation-Service, Portfolio-Analyse, Portfolio-Optimierung |
| Presentation | ~15 | Chart-Rauchtest, UI-Widgets |
| Phase 3 (Integration) | ~75 | Fade für alle Parameter, Copula mit Korrelation, ppf aller 6 Verteilungen, Quality Score, TV/EV, ROIC |

---

## 🤝 Mitwirken

Beiträge sind willkommen! So kannst du mitwirken:

1. **Fork** das Repository
2. **Feature-Branch** erstellen (`git checkout -b feature/mein-feature`)
3. **Änderungen committen** (`git commit -m 'feat: Beschreibung'`)
4. **Branch pushen** (`git push origin feature/mein-feature`)
5. **Pull Request** öffnen

---

## 📄 Lizenz

Dieses Projekt ist unter der [MIT-Lizenz](https://opensource.org/licenses/MIT) lizenziert.

---

<p align="center">
  Erstellt mit ❤️ und viel Statistik
</p>