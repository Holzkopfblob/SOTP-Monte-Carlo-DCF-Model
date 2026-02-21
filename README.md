# 📊 SOTP Monte-Carlo DCF Modell + Portfolio-Optimierung

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-vektorisiert-013243?logo=numpy&logoColor=white)](https://numpy.org/)

> **Zwei unabhängige Streamlit-Apps** für professionelle Unternehmensbewertung und Portfoliostrukturierung — vollständig stochastisch, vektorisiert und interaktiv.

**App 1 – SOTP DCF**: Sum-of-the-Parts Unternehmensbewertung mit Monte-Carlo-Simulation über beliebig viele Geschäftssegmente.

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
- [Mitwirken](#-mitwirken)

---

## ✨ Highlights

| Eigenschaft | Detail |
|---|---|
| **Vektorisierte MC-Engine** | 10.000–500.000 Iterationen via NumPy — keine Python-Loops über Simulationen |
| **6 Wahrscheinlichkeitsverteilungen** | Fest, Normal, Lognormal, Dreieck, Gleichverteilung, PERT |
| **2 Terminal-Value-Methoden** | Gordon Growth Model & Exit-Multiple |
| **7 Portfolio-Optimierungen** | Max Sharpe, Min Volatilität, Risk Parity, Min CVaR, Max Diversification, Multi-Asset Kelly, Gleichgewichtung |
| **Clean Architecture** | 4-Schichten-Architektur (Domain → Application → Infrastructure → Presentation) |
| **Interaktive Charts** | Histogramm+KDE, CDF, Tornado, Waterfall, Efficient Frontier (Plotly) |
| **Excel-Export** | Summary, Assumptions, Raw-Data für externe Audits |
| **Save/Load** | Gesamte Konfiguration als JSON speichern und laden |
| **Eingebettete Erklärungen** | Ausführliche Theorie-Expander mit LaTeX-Formeln direkt in der App |

---

## 📊 SOTP DCF App – Features

### Bewertungsmodell

- **SOTP-Bewertung** – Beliebige Anzahl an Geschäftssegmenten, individuell konfigurierbar
- **FCFF-Ansatz** – Free Cash Flow to Firm über 9 Werttreiber pro Segment:
  - Revenue Growth, EBITDA-Marge, D&A, Steuersatz, CAPEX, NWC-Veränderung, WACC, Terminal Growth, Exit-Multiple
- **Terminal Value** – Gordon Growth Model oder Exit-Multiple pro Segment, mit automatischen Guards (WACC-Floor, TV-Wachstum < WACC)
- **Corporate Bridge** – Holdingkosten (als Perpetuity), Net Debt, verwässerte Aktienanzahl

### Stochastische Simulation

- **6 Verteilungstypen** – Jeder der 9 Parameter kann deterministisch (fest) oder stochastisch über 5 Verteilungen modelliert werden
- **Vektorisierte MC-Engine** – Komplett via NumPy (keine For-Schleifen), reproduzierbar über Seed
- **Sensitivity Analysis** – Spearman-Rangkorrelation identifiziert automatisch die einflussreichsten Werttreiber

### Visualisierung & Export

- **6 interaktive Charts** – EV-Histogramm+KDE, Equity-Histogramm, CDF mit Perzentilen, Preis/Aktie-Histogramm, Tornado-Chart (Top-15 Treiber), SOTP-Waterfall
- **Statistik-Dashboard** – Mean, Median, Std, P5/P25/P75/P95, Min/Max für EV, Equity und Preis/Aktie
- **Excel-Export** – 3 Sheets (Summary & Statistics, Segment Assumptions, Raw Simulation Data bis 100k Zeilen)
- **JSON Save/Load** – Komplette Konfiguration exportieren/importieren
- **Portfolio-App-Link** – μ, σ, Schiefe, Kurtosis und Perzentile als Kopiervorlage für die Portfolio-App

### 4-Tab-Oberfläche

| Tab | Inhalt |
|---|---|
| ⚙️ **Setup** | MC-Iterationen, Seed, Segmentanzahl, Corporate Bridge |
| 🏢 **Segmente** | Pro Segment: Name, Basisumsatz, Prognosejahre, 9 stochastische Werttreiber, TV-Methode |
| 🎲 **Simulation** | Konfigurationsübersicht, Start-Button mit Fortschrittsbalken |
| 📈 **Ergebnisse** | Metriken, Charts, Segment-Details, Excel-Export |

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

Das Projekt folgt **Clean Architecture** Prinzipien mit strikter Schichtentrennung:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Presentation Layer                                                 │
│  app.py (790 LOC) · portfolio_app.py (~1150 LOC)                   │
│  presentation/charts.py · presentation/ui_helpers.py               │
├─────────────────────────────────────────────────────────────────────┤
│  Application Layer                                                  │
│  application/simulation_service.py · application/portfolio_service.py│
├─────────────────────────────────────────────────────────────────────┤
│  Domain Layer (reine Geschäftslogik, kein I/O)                      │
│  domain/models.py · domain/distributions.py · domain/valuation.py  │
├─────────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                               │
│  infrastructure/monte_carlo_engine.py · infrastructure/excel_export.py│
└─────────────────────────────────────────────────────────────────────┘
```

**Kernprinzipien:**

- **Domain Layer** enthält reine Datenstrukturen (`@dataclass`, `Enum`) und Mathematik — kein Framework-Code, kein I/O
- **Infrastructure** kapselt die MC-Engine und Excel-Export
- **Application** orchestriert Use Cases (Simulation starten, Sensitivity berechnen, Portfolio optimieren)
- **Presentation** ist austauschbar — Streamlit-Widgets, Plotly-Charts, UI-Erklärungen

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

wobei $NOPAT = EBITDA \times (1 - t)$ und Revenue exponentiell wächst: $R_t = R_0 \cdot (1+g)^t$

### Terminal Value

**Gordon Growth Model:**
$$TV = \frac{FCFF_T \cdot (1+g)}{WACC - g}$$

**Exit-Multiple:**
$$TV = EBITDA_T \times Multiple$$

### Enterprise Value (DCF)

$$EV = \sum_{t=1}^{T} \frac{FCFF_t}{(1+WACC)^t} + \frac{TV}{(1+WACC)^T}$$

### Equity Value (SOTP Bridge)

$$Equity = \sum_i EV_i - PV(Holdingkosten) - Net\;Debt$$

### Multi-Asset Kelly Criterion

**Einzeltitel:**
$$f^* = \frac{E[R]}{Var(R)}$$

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
- 10 Parameter je Segment mit Verteilungsempfehlungen
- WACC-Ableitung via CAPM + Hamada-Unlever/Re-lever
- Corporate Bridge (Holdingkosten, Net Debt, verwässerte Aktien)
- Plausibilitätschecks und Quellenverzeichnis

Das Ausgabeformat ist direkt in die Streamlit-App übertragbar.

---

## 📁 Projektstruktur

```
SOTP-Monte-Carlo-DCF-Model/
│
├── app.py                          ← Streamlit Entry Point (SOTP DCF, 790 LOC)
├── portfolio_app.py                ← Streamlit Entry Point (Portfolio, ~1150 LOC)
├── requirements.txt                ← Python-Abhängigkeiten
├── README.md                       ← Diese Datei
│
├── domain/                         ← Reine Geschäftslogik (kein I/O)
│   ├── models.py                   ← Dataclasses, Enums (150 LOC)
│   ├── distributions.py            ← 6 Verteilungsklassen + Factory (188 LOC)
│   └── valuation.py                ← FCFF, TV, DCF-Berechnung (164 LOC)
│
├── infrastructure/                 ← Technische Infrastruktur
│   ├── monte_carlo_engine.py       ← Vektorisierte MC-Engine (131 LOC)
│   └── excel_export.py             ← xlsx-Report mit 3 Sheets (147 LOC)
│
├── application/                    ← Use-Case-Orchestrierung
│   ├── simulation_service.py       ← Simulation + Sensitivity (75 LOC)
│   └── portfolio_service.py        ← 7 Optimierungen, Stress-Tests, Cluster-Korrelation (~780 LOC)
│
├── presentation/                   ← UI & Visualisierung
│   ├── ui_helpers.py               ← Streamlit-Widgets & Erklärungen (424 LOC)
│   └── charts.py                   ← 13 Plotly-Chartgeneratoren (~510 LOC)
│
└── prompts/
    └── sotp_research_prompt.md     ← LLM-Research-Prompt (331 LOC)
```

**Gesamtumfang: ~4.500+ Zeilen Code**

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