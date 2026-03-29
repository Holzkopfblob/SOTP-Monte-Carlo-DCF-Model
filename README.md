# 📊 SOTP Monte-Carlo DCF Modell (Web-App)

Diese README ist für Nutzer der gehosteten Anwendung:

👉 https://sotp-monte-carlo-dcf-model.streamlit.app/

Sie erklärt **was** die App macht, **wie** du sie effizient nutzt und **warum** der Research Prompt der wichtigste Input-Baustein ist.

## Was ist das?

Die App bewertet Unternehmen mit einem stochastischen **Sum-of-the-Parts-DCF** (Monte Carlo):

- Mehrere Segmente mit eigenen Treiber-Verteilungen (Wachstum, Marge, CAPEX, WACC usw.)
- Korrelationen zwischen Segmenten und optional innerhalb von Segmenten
- Ergebnis nicht als einzelner Wert, sondern als Verteilung (inkl. Risiko-/Qualitätsdiagnostik)

## Warum der Research Prompt?

Die Ergebnisqualität hängt direkt von der Qualität deiner Eingaben ab.

Der Prompt in [prompts/sotp_research_prompt.md](prompts/sotp_research_prompt.md) liefert dafür einen strukturierten Standard:

- Vollständige Parameter-Checkliste für alle Segmente
- Einheitliches Format für Verteilungen, Begründungen und Quellen
- Strikte Zitationslogik (`[Q1]`, `[Q2]`, …)
- Wizard-taugliche Ausgabe-Reihenfolge

Kurz: Der Prompt sorgt dafür, dass du aus „Meinung“ belastbare, nachvollziehbare Inputs machst.

## Wie nutzen? (empfohlener Ablauf)

1. **Research vorbereiten**
	- Kopiere den Prompt aus [prompts/sotp_research_prompt.md](prompts/sotp_research_prompt.md)
	- Ersetze `[UNTERNEHMEN]`
	- Führe ihn mit einem Modell/Tool mit Web- und Datenzugriff aus

2. **Ergebnisse in den Wizard übertragen**
	- **⚙️ Setup:** Iterationen, Seed, Sampling, Corporate Bridge
	- **🏢 Segmente:** Segment für Segment die Verteilungen eintragen
	- **🎲 Simulation:** Lauf starten
	- **📈 Ergebnisse:** Verteilungen, Risiko, Treiber und Qualität interpretieren

3. **Szenarien vergleichen**
	- Konservative vs. Base vs. Upside-Annahmen als separate Durchläufe
	- Ergebnisse über Quantile statt nur über Mittelwerte vergleichen

## Was im Prompt besonders wichtig ist

- **Einheiten:** Prozent als Prozentwerte (z. B. `5.0` für 5 %), Beträge in Mio.
- **Verteilungen:** Unsicherheit realistisch modellieren (nicht alles auf „Fest“)
- **Quellen:** Jede zentrale Annahme belegen und zitieren
- **Konsistenzchecks:** Plausibilitätsprüfungen im Prompt ernst nehmen

## Ergebnis-Interpretation (praktisch)

- Nutze nicht nur den erwarteten Wert, sondern v. a. Quantile (z. B. P5/P50/P95)
- Prüfe, welche Treiber Sensitivität und Tail-Risiko dominieren
- Achte auf Qualitätsindikatoren (Konvergenz, TV/EV-Anteil, Robustheit)

## Troubleshooting (Web-App)

- **Keine Ergebnisse sichtbar:** Im Schritt **🎲 Simulation** den Lauf starten.
- **Werte wirken unplausibel:** Quellen, Einheiten und Verteilungsparameter aus dem Prompt gegenprüfen.
- **Instabile Aussagen:** Mehr Iterationen wählen und Annahmenbandbreiten prüfen.

## Hinweis

Diese README beschreibt bewusst den Nutzer-Workflow der gehosteten App. Technische Setup-/Entwicklerhinweise sind hier nicht im Fokus.
