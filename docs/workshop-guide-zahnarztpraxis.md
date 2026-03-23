# Workshop-Guide: SOTP Monte-Carlo DCF fuer eine Zahnarztpraxis

## Zweck und Einsatz
Dieses Dokument ist ein Moderationsskript fuer einen 75-90-Minuten-Workshop. Es fuehrt Schritt fuer Schritt durch die DCF/SOTP-App und erklaert alle zentralen Methoden in alltagsnaher Sprache.

## Zielgruppe
- Zahnaerztinnen und Zahnaerzte ohne tiefe Finance-Vorkenntnisse
- Praxisinhaberinnen und Praxisinhaber vor Kauf, Verkauf oder Ausbau
- Berater, die ein laienverstaendliches Bewertungsnarrativ brauchen

## Lernziele
Nach dem Workshop koennen Teilnehmende:
1. Daten fuer eine praxisnahe Bewertung vorbereiten,
2. im Tool Setup, Segmente, Simulation und Ergebnisdeutung sauber durchlaufen,
3. P50, P5, CVaR und Sensitivitaeten fuer Entscheidungen nutzen,
4. den Unterschied zwischen Geschaeftswert (EV) und Eigentuemerwert (Equity) sicher erklaeren.

## Aufbau des Workshops
- Dauer: 75-90 Minuten
- Format: Live-Demo plus kurze Methodenboxen
- Umfang: Nur DCF/SOTP-App, kein Portfolio-Teil

## Agenda mit Zeitbudget
| Block | Dauer | Ziel | Ergebnis |
|---|---:|---|---|
| Einstieg und Erwartungsabgleich | 10 Min | Problemrahmen setzen | Teilnehmende verstehen den Nutzen einer Verteilung statt Einzelwert |
| Fallprofil und Datengeruest | 10 Min | Beispiel verankern | Gemeinsames Bild der Zahnarztpraxis |
| Tool-Setup und Segmenteingaben | 25 Min | Modell konfigurieren | Simulationsfaehige Eingaben fuer 3 Segmente |
| Simulationslauf und Ergebnisdeutung | 25 Min | Charts korrekt lesen | Handlungsfaehige Interpretation (P50, P5, Treiber) |
| Entscheidung, Q&A, Takeaways | 15 Min | Transfer in Praxis | Klare Kauf-/Verkaufslogik in 4 Schritten |

## Kapitel 1 - Einstieg: Warum ein Simulator statt einer Punktzahl?
### Kernbotschaft
Eine Praxis hat nicht nur einen Wertpunkt, sondern ein Wertspektrum. Unsichere Annahmen (Wachstum, Marge, Kapitalkosten) erzeugen unterschiedliche moegliche Zukuenfte.

### Moderationsskript
"Heute treffen wir keine Entscheidung auf Basis einer einzigen Zahl, sondern auf Basis von Wahrscheinlichkeiten. So sehen wir realistisch, was wahrscheinlich ist und was im schlechten Fall passieren kann."

### Leitfragen an die Gruppe
- Welche 2 Faktoren beeinflussen den Praxiswert am staerksten?
- Welche Unsicherheit ist aus Ihrer Sicht am schwersten zu schaetzen?

### Soll-Ergebnis
Teilnehmende akzeptieren, dass Bandbreiten fuer Entscheidungen belastbarer sind als Punktwerte.

## Kapitel 2 - Fallbeispiel Zahnarztpraxis
### Praxisprofil (konsistent ueber alle Kapitel)
- Standort: Grossstadt, etablierte Lage
- Team: 3 Behandler, 6 Assistenz/Verwaltung
- Umsatz aktuell: 2.0 Mio EUR
- EBITDA-Marge aktuell: 34 Prozent
- Nettoverschuldung: 0.45 Mio EUR

### Segmentierung fuer SOTP
- Segment A: Allgemeine Zahnheilkunde und Prophylaxe (70 Prozent Umsatzanteil)
- Segment B: Implantologie und Chirurgie (25 Prozent Umsatzanteil)
- Segment C: Zusatzleistungen/Laborvermittlung (5 Prozent Umsatzanteil)

### Didaktischer Hinweis
Alle Zahlen sind plausible Beispielwerte zur Erlaeuterung, keine Branchenreferenz.

## Kapitel 3 - Datenbedarf vor der Eingabe
### Minimum-Datensatz
- Umsatz je Segment (letztes Jahr)
- EBITDA-Marge je Segment
- Steuerquote
- CAPEX-Quote
- NWC-Aenderung (vereinfachter Prozentansatz)
- WACC-Annahme
- Langfristiges Wachstum

### Plausibilitaetsregeln fuer Laien
- Bei Unsicherheit immer Verteilung statt Fixwert.
- Kritische Treiber zuerst: Wachstum, Marge, WACC.
- Lieber realistische Korridore als scheinbar genaue Einzelfestlegung.

## Kapitel 4 - Setup-Tab: technische Parameter setzen
### Ziel
Reproduzierbare und stabile Simulationen vorbereiten.

### Demo-Schritte
1. App starten mit `streamlit run app.py`.
2. Setup-Tab oeffnen.
3. Iterationen auf 10000 setzen.
4. Seed auf 42 setzen.
5. Sampling auf Pseudo-Random lassen, spaeter Sobol vergleichen.
6. Segmentanzahl auf 3 setzen.
7. Corporate-Bridge aktivieren.

### Erklaerung in 30 Sekunden
- Mehr Iterationen reduzieren Zufallsrauschen.
- Seed macht Ergebnisse wiederholbar.
- Sobol kann schneller stabile Perzentile liefern.

### Soll-Ergebnis
Technisch gueltige Konfiguration ohne Inkonsistenzen.

## Kapitel 5 - Segment-Tab: Treiber als Verteilungen eingeben
### Ziel
Werttreiber pro Segment realistisch als Unsicherheitskorridor modellieren.

### Workshop-Startwerte
- Wachstum Segment A (PERT): Min 1.0, Mode 3.0, Max 5.0
- Wachstum Segment B (PERT): Min 2.0, Mode 6.0, Max 10.0
- Wachstum Segment C (PERT): Min -1.0, Mode 2.0, Max 6.0
- EBITDA-Marge A (Normal): Mittel 30.0, Std 2.0
- EBITDA-Marge B (Normal): Mittel 42.0, Std 3.0
- EBITDA-Marge C (Normal): Mittel 18.0, Std 2.0
- WACC (Normal): Mittel 8.0, Std 1.0
- Terminal Growth (Fix): 2.0

### Moderationsskript fuer den Eingabefluss
1. Segment A vollstaendig eingeben und erklaeren.
2. Segment B schneller spiegeln, Fokus auf abweichende Treiber.
3. Segment C als kleineres Beimischungssegment behandeln.
4. Vor Start eine Plausibilitaetsrunde mit der Gruppe machen.

### Methodenbox - FCFF in einfach
Formel:
$FCFF = NOPAT + D\&A - CAPEX - \Delta NWC$

Interpretation:
"FCFF ist das frei verfuegbare Geld aus dem operativen Geschaeft nach notwendigen Investitionen und Betriebskapitalbindung."

## Kapitel 6 - Verteilungen ohne Mathematikstress
### Kurzleitfaden
- Fixed: nur bei sehr hoher Sicherheit
- Normal: symmetrische Schwankung um den Mittelwert
- LogNormal: nach unten begrenzt, nach oben offener
- PERT: Min/Mode/Max aus Expertensicht
- Triangular: einfache Alternative zu PERT
- Uniform: alle Werte im Intervall gleich wahrscheinlich

### Entscheidungsregel fuer Workshops
Wenn keine robuste Historie vorliegt, mit PERT starten und erst in Runde 2 verfeinern.

### Methodenbox - WACC in einfach
"WACC ist der Risikopreis des eingesetzten Kapitals. Steigt WACC, sinkt der heutige Wert der gleichen Zukunfts-Cashflows."

## Kapitel 7 - Simulation-Tab: Lauf starten
### Demo-Schritte
1. Simulation-Tab oeffnen.
2. Konfiguration gegenpruefen.
3. Lauf starten.
4. Laufzeit und Fortschritt kurz kommentieren.

### Methodenbox - Monte Carlo in einfach
"Der Rechner zieht viele moegliche Zukunftspfade, bewertet jeden Pfad und baut daraus eine Verteilung des Praxiswerts."

### Soll-Ergebnis
Valider Ergebnisdatensatz mit EV-/Equity-Verteilung und Risiko-Kennzahlen.

## Kapitel 8 - Ergebnisse lesen wie ein Entscheider
### Reihenfolge in der Ergebnisbesprechung
1. P50 als typischer Erwartungswert
2. P5/P95 als Risiko- und Chancenband
3. Histogramm und CDF fuer Wahrscheinlichkeiten
4. Tornado fuer die Top-Werttreiber

### Leitfragen
- Wie gross ist der Abstand zwischen gefordertem Preis und P50?
- Wie tief liegt P5 und ist diese Downside tragbar?
- Welche zwei Treiber dominieren den Wert?

### Methodenbox - DCF in einfach
"Zukuenftige freie Cashflows werden auf heute abgezinst. Hoeheres Risiko bedeutet staerkere Abzinsung und damit niedrigeren heutigen Wert."

## Kapitel 9 - Tail Risk und Sensitivitaet
### Tail Risk
- P5: konservatives Downside-Perzentil
- CVaR: durchschnittlicher Wert in den schlechtesten Faellen

### Sensitivitaet
Das Tornado-Diagramm zeigt, welche Annahmen den groessten Einfluss auf den Equity-Wert haben.

### Interpretation fuer die Praxis
Wenn WACC und Wachstum dominieren, ist die wichtigste Aufgabe nicht mehr Rechnen, sondern diese beiden Annahmen besser zu validieren.

## Kapitel 10 - EV zu Equity: die Bridge
### Methodenbox - Equity Bridge in einfach
Formel:
$Equity = EV - NetDebt - Minderheiten - Pensionen + NonOp + Beteiligungen$

Interpretation:
"Vom Wert des operativen Geschaefts gehen finanzielle Verpflichtungen ab; zusaetzliche nicht-operative Werte werden addiert."

### Demo-Schritt
Nettoverschuldung im Bridge-Teil variieren und den direkten Effekt auf Equity-Wert zeigen.

## Kapitel 11 - Entscheidung fuer Praxisinhaber
### 4-Schritte-Framework
1. Preischeck: Liegt der Kaufpreis unter P50?
2. Sicherheitscheck: Gibt es genug Abstand zu P5?
3. Treibercheck: Sind Top-Sensitivitaeten sachlich plausibel?
4. Steuerbarkeitscheck: Welche Treiber sind operativ beeinflussbar?

### Beispielhafte Entscheidungsaussage
"Bei Kaufpreis 2.4 Mio EUR und P50 von 2.8 Mio EUR besteht eine moderate Sicherheitsmarge. Hauptunsicherheit bleibt der Block aus WACC und Wachstum."

## Kapitel 12 - Kompletter Base/Bull/Bear-Durchlauf
### Base Case
- Segmentannahmen wie in Kapitel 5
- Ziel: Referenzverteilung mit moderater Unsicherheit

### Bull Case (eine Runde)
- Wachstum Segment B um +2 Prozentpunkte anheben
- EBITDA-Marge Segment A und B je +1 Prozentpunkt
- Erwartung: P50 steigt, rechte Verteilungsschulter wird breiter

### Bear Case (eine Runde)
- WACC-Mittel von 8.0 auf 9.0 anheben
- Wachstum Segment A und C jeweils -1 Prozentpunkt
- Erwartung: P5 und CVaR verschlechtern sich sichtbar

### Lerneffekt
Nicht das einzelne Ergebnis ist entscheidend, sondern wie empfindlich die Bewertung auf realistische Stressaenderungen reagiert.

## Kapitel 13 - FAQ und Glossar
### FAQ
- Warum ist P5 deutlich niedriger als P50?
  - Weil mehrere negative Treiber gemeinsam auftreten koennen.
- Soll ich exakte Werte oder Bandbreiten nutzen?
  - Fuer kritische Treiber immer Bandbreiten.
- Warum reagiert der Wert stark auf WACC?
  - Weil WACC alle Zukunfts-Cashflows direkt abzinst.

### Glossar
- DCF: Discounted Cash Flow
- FCFF: Free Cash Flow to Firm
- WACC: gewichtete Kapitalkosten
- SOTP: Sum of the Parts
- P50: Median der Verteilung
- P5: konservatives Downside-Perzentil
- CVaR: Mittelwert der schlechtesten Szenarien

## Anhang A - Moderator-Checkliste vor dem Termin
- App laeuft lokal stabil
- Beispielwerte vorbereitet
- Reihenfolge der Charts festgelegt
- Kernfragen pro Kapitel sichtbar notiert
- Abschlussslide mit 3 Kernbotschaften bereit

## Anhang B - Soll-Outputs fuer die Dokumentation
- Screenshot 1: Setup mit 10000 Iterationen und Seed 42
- Screenshot 2: Segmenteingaben fuer A/B/C
- Screenshot 3: Ergebnis-Histogramm mit P5/P50/P95
- Screenshot 4: Tornado mit Top-5-Treibern
- Screenshot 5: Equity-Bridge mit gezeigter NetDebt-Wirkung

## Anhang C - Qualitaetskriterien der fertigen Fassung
- Ein fachfremder Leser kann die Schritte ohne Zusatzhilfe ausfuehren.
- Jede Methodenbox bleibt in maximal 3-4 Saetzen.
- Zahlen und Segmentlogik bleiben ueber alle Kapitel konsistent.
- Modellannahmen und Fakten sind sprachlich klar getrennt.
