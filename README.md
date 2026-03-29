# 📊 SOTP Monte-Carlo DCF Modell

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-vektorisiert-013243?logo=numpy&logoColor=white)](https://numpy.org/)

Interaktive Streamlit-Anwendung für stochastische Sum-of-the-Parts-Unternehmensbewertung mit Monte-Carlo-Simulation.

## ✅ Voraussetzungen

- Python 3.10+
- `pip` (oder venv/conda)
- Empfohlen: virtuelle Umgebung

## ✨ Features

- Vektorisierte MC-Engine (NumPy) mit Pseudo-Random und Sobol-Sampling
- Multi-Segment-SOTP mit FCFF-Ansatz und stochastischen Treibern
- Universal-Fade-Modell für FCFF-Parameter
- Cross-Segment- und Intra-Segment-Korrelation (Gauss-Copula)
- Tail-Risk, Economic-Profit, Margin-of-Safety, Conditional Sensitivity
- TV/EV-, ROIC- und Konvergenz-Qualitätsdiagnostik
- Excel-Export und JSON Save/Load

## 🚀 Schnellstart

```bash
git clone https://github.com/Holzkopfblob/SOTP-Monte-Carlo-DCF-Model.git
cd SOTP-Monte-Carlo-DCF-Model
pip install -r requirements.txt
streamlit run app.py
```

### Windows (PowerShell, empfohlen)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 🧭 Nutzung (Wizard-Flow)

1. **Setup:** Simulation, Segmentanzahl, Bridge und Segment-Korrelation festlegen.
2. **Segmente:** Treiber, TV-Methodik, Fade-Optionen und intra-segment Korrelation pflegen.
3. **Simulation:** Lauf starten und Ergebnisse in Session laden.
4. **Ergebnisse:** Überblick, Risiko, Treiber, Qualität, Detail-Analysen und Export.

## 💾 Konfiguration speichern/laden

- Export/Import erfolgt über die Sidebar als JSON.
- Aktuelles Format nutzt `schema_version: 2` mit `ui_state`.
- Legacy-Dateien mit altem `setup`-Root werden weiterhin eingelesen.

## 🏗 Architektur

- `presentation/`: Streamlit-Seiten, UI-Helfer, Charts
- `application/`: Use-Case-Orchestrierung
- `domain/`: Modelle, Verteilungen, Bewertungslogik
- `infrastructure/`: MC-Engine, Export, Konfig-IO

## 📁 Projektstruktur

```text
SOTP-Monte-Carlo-DCF-Model/
├── app.py
├── requirements.txt
├── pytest.ini
├── README.md
├── application/
├── domain/
├── infrastructure/
├── presentation/
├── prompts/
└── tests/
```

## 🧪 Tests

```bash
pytest -q
```

## 🛠 Troubleshooting

- **`streamlit`/`pytest` nicht gefunden:** virtuelle Umgebung aktivieren.
- **Leere Ergebnisseite:** zuerst im Schritt **Simulation** ausführen.
- **Langsame Detailcharts:** in Ergebnisse unter „Erweiterte Detailcharts (lazy)" nur bei Bedarf aktivieren.

## 📄 Lizenz

[MIT-Lizenz](https://opensource.org/licenses/MIT)
