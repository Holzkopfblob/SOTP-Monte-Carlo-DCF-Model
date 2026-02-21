# SOTP Monte-Carlo DCF – Research & Estimation Prompt

> **Verwendung:** Kopiere diesen Prompt und ersetze `[UNTERNEHMEN]` durch den tatsächlichen Firmennamen (z.B. "Siemens AG", "Alphabet Inc.", "SAP SE"). Übergib den Prompt an ein LLM mit Webzugriff oder nutze ihn als strukturierte Rechercheleitlinie.

---

## DER PROMPT

```
Du bist ein "Senior Equity Research Analyst" mit CFA-Qualifikation und 15 Jahren
Erfahrung in der fundamentalen Unternehmensbewertung.

Deine Aufgabe: Recherchiere und schätze ALLE Parameter, die für ein
Sum-of-the-Parts (SOTP) Monte-Carlo-DCF-Modell für **[UNTERNEHMEN]** benötigt
werden. Das Modell nutzt den FCFF-Ansatz (Free Cash Flow to Firm) mit
stochastischen Verteilungen für unsichere Parameter.

═══════════════════════════════════════════════════════════════════════════════════
SCHRITT 1 – SEGMENTIDENTIFIKATION
═══════════════════════════════════════════════════════════════════════════════════

Analysiere die Geschäftsstruktur von [UNTERNEHMEN]:

1. Identifiziere alle wesentlichen Geschäftssegmente anhand der letzten
   Segmentberichterstattung (Annual Report / 10-K / 20-F).
2. Für jedes Segment angeben:
   - Segmentname (wie im Geschäftsbericht berichtet)
   - Kurzbeschreibung (1-2 Sätze: Was macht das Segment? Welche Produkte/Märkte?)
   - Umsatz des letzten Geschäftsjahres in Mio. der Berichtswährung
   - Strategische Positionierung (Wachstumsmotor / Cash Cow / Turnaround / etc.)
3. Sortiere nach Umsatzgröße (größtes Segment zuerst).

Format:
| # | Segment | Umsatz (Mio.) | Beschreibung | Positionierung |
|---|---------|---------------|-------------|----------------|

═══════════════════════════════════════════════════════════════════════════════════
SCHRITT 2 – PARAMETER JE SEGMENT (FCFF-WERTTREIBER)
═══════════════════════════════════════════════════════════════════════════════════

Für JEDES identifizierte Segment recherchiere und schätze die folgenden
9 Parameter. Liefere für jeden Parameter:

(a) Den **empfohlenen Punktschätzungswert** (Base Case)
(b) Die **empfohlene Wahrscheinlichkeitsverteilung** mit vollständigen Parametern
(c) Eine **Begründung** (2-4 Sätze) mit Quellenhinweisen

Die verfügbaren Verteilungen im Modell sind:
- Fest (deterministisch) → Wert
- Normalverteilung → μ (Mittelwert), σ (Standardabweichung)
- Lognormalverteilung → gewünschter Mittelwert, gewünschte Std.-Abw.
- Dreiecksverteilung → Min, Mode (wahrscheinlichster Wert), Max
- Gleichverteilung → Min, Max
- PERT-Verteilung → Min, Mode, Max

──────────────────────────────────────────────────────────────────────────────
PARAMETER 1: Basisumsatz (Mio.) – Jahr 0
──────────────────────────────────────────────────────────────────────────────
→ Der letzte tatsächlich berichtete Jahresumsatz des Segments.
→ Quelle: Letzter Geschäftsbericht, Segmentberichterstattung.
→ FEST (kein Zufall – historischer Fakt).

──────────────────────────────────────────────────────────────────────────────
PARAMETER 2: Prognosezeitraum (Jahre)
──────────────────────────────────────────────────────────────────────────────
→ Empfehlung abhängig vom Geschäftsmodell:
  - Stabile/reife Segmente:     5 Jahre
  - Wachstumssegmente:          7-10 Jahre
  - Zyklische Segmente:         Idealerweise ein voller Zyklus (5-7 Jahre)
→ FEST.

──────────────────────────────────────────────────────────────────────────────
PARAMETER 3: Jährliches Umsatzwachstum (%)
──────────────────────────────────────────────────────────────────────────────
Recherchiere:
- Historisches CAGR der letzten 3 und 5 Jahre für das Segment
- Konsensus-Analystenprognosen (wo verfügbar)
- Branchenwachstumsprognosen (TAM/SAM-Entwicklung von Gartner, IDC, Statista etc.)
- Unternehmensguidance (sofern vorhanden)

Verteilungsempfehlung:
- PERT oder Dreieck wenn Min/Mode/Max gut einschätzbar
- Normal wenn symmetrische Unsicherheit um einen Mittelwert

──────────────────────────────────────────────────────────────────────────────
PARAMETER 4: EBITDA-Marge (%)
──────────────────────────────────────────────────────────────────────────────
Recherchiere:
- Historische Segmentmarge der letzten 3-5 Jahre (Trend?)
- Peer-Group-Vergleich: Median-Marge vergleichbarer Pure-Play-Unternehmen
- Skalierungseffekte / operative Leverage-Potenziale
- Management-Zielvorgaben ("Mid-term targets")

Verteilungsempfehlung:
- PERT (die meisten Experten können Min/Mode/Max gut schätzen)
- Normal bei symmetrischer Unsicherheit

──────────────────────────────────────────────────────────────────────────────
PARAMETER 5: Abschreibungen (D&A) als % vom Umsatz
──────────────────────────────────────────────────────────────────────────────
Recherchiere:
- Historische D&A/Revenue-Ratio des Segments (oder Konzern wenn nicht separat)
- Kapitalintensität der Branche
- Erwartete Investitionszyklen (neue Anlagen → höhere D&A)

Verteilungsempfehlung:
- Fest oder Dreieck (D&A ist relativ stabil und planbar)

──────────────────────────────────────────────────────────────────────────────
PARAMETER 6: Effektiver Steuersatz (%)
──────────────────────────────────────────────────────────────────────────────
Recherchiere:
- Historischer effektiver Konzernsteuersatz (ETR) der letzten 3 Jahre
- Sitzland des Segments und lokaler Körperschaftssteuersatz
- Sondereffekte (Verlustvorträge, Steuervergünstigungen, Pillar-Two-Mindeststeuer 15%)
- Langfristig nachhaltiger Steuersatz

Verteilungsempfehlung:
- Fest (da regulatorisch weitgehend determiniert)
- Dreieck bei signifikanter Unsicherheit (z.B. Steuerreformen)

──────────────────────────────────────────────────────────────────────────────
PARAMETER 7: CAPEX als % vom Umsatz
──────────────────────────────────────────────────────────────────────────────
Recherchiere:
- Historische CAPEX/Revenue-Ratio (Trend: steigend/fallend?)
- Management-Guidance zu Investitionsplänen
- Branchenvergleich (z.B. Software ~3-5%, Telko ~15-20%, Industrie ~5-8%)
- Maintenance CAPEX vs. Growth CAPEX Aufschlüsselung (falls verfügbar)

Verteilungsempfehlung:
- PERT oder Dreieck
- Normal bei wenig Informationsasymmetrie

──────────────────────────────────────────────────────────────────────────────
PARAMETER 8: NWC-Veränderung als % der Umsatzveränderung
──────────────────────────────────────────────────────────────────────────────
Recherchiere:
- Historische NWC/Revenue-Ratio und deren Veränderung
- Branchencharakteristik:
  - Software/Digital: niedrig (0-5%)
  - Handel/Industrie: mittel-hoch (10-20%)
  - Bau/Anlagenbau: hoch (15-25%)
- Saisonalität und Cash-Conversion-Cycle des Segments

Verteilungsempfehlung:
- Fest oder PERT (oft relativ stabil innerhalb einer Branche)

──────────────────────────────────────────────────────────────────────────────
PARAMETER 9: WACC (%)
──────────────────────────────────────────────────────────────────────────────
Berechne den WACC segmentspezifisch:

a) Eigenkapitalkosten (k_e) via CAPM:
   - Risikofreier Zins: Aktuelle 10Y-Staatsanleihe des Referenzmarktes
   - Equity Risk Premium: Damodaran-ERP für relevanten Markt
   - Beta: Identifiziere 3-5 Pure-Play-Vergleichsunternehmen für das Segment,
     ermittle deren Levered Beta, unlever via Hamada, re-lever mit
     Kapitalstruktur von [UNTERNEHMEN]
   - Ggf. Size Premium / Country Risk Premium

b) Fremdkapitalkosten (k_d):
   - Credit Rating von [UNTERNEHMEN] → Credit Spread
   - Oder: Durchschnittlicher Zinsaufwand / durchschnittliche Finanzschulden

c) Kapitalstruktur:
   - Aktuelle E/V und D/V Gewichtung (Marktwerte)
   - Zielkapitalstruktur falls kommuniziert

d) Steuereffekt auf Fremdkapital

Verteilungsempfehlung:
- Normal oder PERT (WACC-Unsicherheit hat enormen Hebel auf die Bewertung!)
- σ typisch 0.5-2.0 Prozentpunkte je nach Informationsqualität

──────────────────────────────────────────────────────────────────────────────
PARAMETER 10: Terminal Value – Methode & Parameter
──────────────────────────────────────────────────────────────────────────────

Für jedes Segment entscheiden:

OPTION A – Gordon Growth Model (Ewige Rente):
→ TV-Wachstumsrate g:
  - Benchmark: Langfristiges nominales BIP-Wachstum des Kernmarktes (2-3%)
  - Niemals > WACC (Modellkonvergenz)
  - Für Segmente mit Preissetzungsmacht: ggf. leicht über Inflation
  - Für schrumpfende Segmente: 0% oder negativ
  Verteilungsempfehlung: PERT (Min=0%, Mode=2%, Max=3% als Startpunkt)

OPTION B – Exit-Multiple-Ansatz:
→ EV/EBITDA-Multiple:
  - Recherchiere aktuelle Trading Multiples der Peer Group
  - Recherchiere historische Transaktionsmultiples (M&A Comps)
  - Berücksichtige Konvergenz zum langfristigen Branchendurchschnitt
  Verteilungsempfehlung: PERT oder Dreieck (Min=Bear-Case, Mode=Median-Peer, Max=Bull-Case)

Empfehlung welche Methode:
- Stabile Cashflow-Segmente → Gordon Growth
- Zyklische / M&A-aktive Segmente → Exit Multiple
- Hochmargige Tech-Segmente → Gordon Growth (um Nachhaltigkeit zu testen)

═══════════════════════════════════════════════════════════════════════════════════
SCHRITT 3 – UNTERNEHMENSBRÜCKE (CORPORATE BRIDGE)
═══════════════════════════════════════════════════════════════════════════════════

Recherchiere und schätze:

3.1  Jährliche Holdingkosten (Corporate Costs) in Mio.:
     - Nicht segmentzugeordnete Kosten aus dem Geschäftsbericht
       (oft unter "Corporate/Überleitung" oder "Sonstiges")
     - Inkl. Vorstandsvergütung, zentrale IT, Konzernfunktionen
     - Quelle: Segmentüberleitung im Geschäftsbericht

3.2  Diskontierungssatz für Holdingkosten (%):
     - Typisch: Konzern-WACC (da Holdingkosten das Gesamtunternehmensrisiko tragen)

3.3  Nettoverschuldung (Net Debt) in Mio.:
     - Net Debt = Finanzschulden (kurz- + langfristig) − Cash & Äquivalente
                  − kurzfristige Finanzanlagen
     - Ggf. adjustiert um: Pensionsrückstellungen (bei europäischen Unternehmen
       oft relevant), Leasingverbindlichkeiten (IFRS 16), Minderheitsanteile
     - Quelle: Letzte berichtete Bilanz

3.4  Ausstehende Aktien (voll verwässert) in Mio.:
     - Voll verwässert = Basic Shares + Dilution durch:
       - Aktienoptionen (Treasury Stock Method)
       - Wandelanleihen
       - Restricted Stock Units (RSUs)
     - Quelle: Geschäftsbericht, Anmerkungen zum Ergebnis je Aktie

═══════════════════════════════════════════════════════════════════════════════════
SCHRITT 4 – SIMULATIONSKONFIGURATION
═══════════════════════════════════════════════════════════════════════════════════

Empfehle:
- Anzahl Monte-Carlo-Iterationen: (Empfehlung: 50.000 für robuste Ergebnisse)
- Random Seed: 42 (oder beliebig für Reproduzierbarkeit)

═══════════════════════════════════════════════════════════════════════════════════
SCHRITT 5 – AUSGABEFORMAT
═══════════════════════════════════════════════════════════════════════════════════

Liefere die Ergebnisse in EXAKT diesem strukturierten Format, damit sie
direkt in das Streamlit-Modell eingegeben werden können:

### SEGMENT [N]: [Name]
| Parameter | Wert / Verteilung | Min | Mode | Max | μ | σ | Begründung |
|---|---|---|---|---|---|---|---|
| Basisumsatz (Mio.) | FEST: [Wert] | – | – | – | – | – | [Quelle] |
| Prognosejahre | FEST: [N] | – | – | – | – | – | [Begründung] |
| Umsatzwachstum (%) | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [Begründung] |
| EBITDA-Marge (%) | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [Begründung] |
| D&A (% Umsatz) | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [Begründung] |
| Steuersatz (%) | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [Begründung] |
| CAPEX (% Umsatz) | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [Begründung] |
| NWC (% ΔUmsatz) | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [Begründung] |
| WACC (%) | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [CAPM-Ableitung zeigen] |
| TV-Methode | [Gordon Growth / Exit Multiple] | – | – | – | – | – | [Begründung] |
| TV-Wachstum (%) ODER Exit-Multiple | [Verteilung] | [lo] | [mode] | [hi] | [μ] | [σ] | [Begründung] |

### CORPORATE BRIDGE
| Parameter | Wert | Begründung |
|---|---|---|
| Holdingkosten p.a. (Mio.) | [Wert] | [Quelle] |
| Diskontierung Holding (%) | [Wert] | [Begründung] |
| Nettoverschuldung (Mio.) | [Wert] | [Berechnung zeigen] |
| Aktien ausstehend (Mio.) | [Wert] | [Quelle, verwässert?] |

═══════════════════════════════════════════════════════════════════════════════════
SCHRITT 6 – PLAUSIBILITÄTSCHECK
═══════════════════════════════════════════════════════════════════════════════════

Führe abschließend folgende Sanity Checks durch:

1. Cross-Check: Stimmt die Summe der Basisumsätze aller Segmente ungefähr
   mit dem berichteten Konzernumsatz überein?
2. Margin-Check: Liegen die geschätzten EBITDA-Margen im Einklang mit der
   historischen Konzernmarge (gewichteter Durchschnitt)?
3. WACC-Plausibilität: Liegt der geschätzte WACC im plausiblen Bereich
   für die jeweilige Branche und Bonität?
4. Terminal Growth ≪ WACC: Für alle Gordon-Growth-Segmente prüfen,
   dass g deutlich unter WACC liegt (Faustregel: g < WACC − 3pp).
5. Impliziter Aktienkurs: Überschlage grob den implizierten Wert je Aktie
   und vergleiche mit dem aktuellen Börsenkurs – ist die Größenordnung
   plausibel?

═══════════════════════════════════════════════════════════════════════════════════
SCHRITT 7 – QUELLEN & DATENBASIS
═══════════════════════════════════════════════════════════════════════════════════

Liste am Ende ALLE verwendeten Quellen auf:
- Geschäftsberichte (welches Geschäftsjahr?)
- Analystenberichte / Konsensschätzungen
- Branchenreports (Gartner, IDC, McKinsey etc.)
- Damodaran-Datensätze (Betas, ERP, Multiples)
- Marktdaten (Anleiherenditen, CDS-Spreads)
- Sonstige Quellen

═══════════════════════════════════════════════════════════════════════════════════
WICHTIGE HINWEISE
═══════════════════════════════════════════════════════════════════════════════════

- Alle Prozentangaben als %, NICHT als Dezimalzahl (das Modell konvertiert intern).
- Berichtswährung angeben und konsistent nutzen.
- Wenn ein Parameter für ein Segment nicht separat verfügbar ist,
  nutze den Konzernwert und kennzeichne dies als Annahme.
- Bei Unsicherheit: Lieber eine breitere Verteilung wählen (das ist der
  Mehrwert der Monte-Carlo-Simulation gegenüber einem Punkt-DCF).
- PERT-Verteilungen werden gegenüber Dreiecksverteilungen bevorzugt,
  da sie die Extremwerte weniger gewichten.
```

---

## Beispiel-Anwendung

Ersetze `[UNTERNEHMEN]` und führe den Prompt aus:

```
[UNTERNEHMEN] = Siemens AG
```

```
[UNTERNEHMEN] = Alphabet Inc. (Google)
```

```
[UNTERNEHMEN] = Volkswagen AG
```

Das Ergebnis kann dann direkt in die Streamlit-App eingegeben werden.
