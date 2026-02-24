# SOTP Monte-Carlo DCF – Research & Estimation Prompt

> **Anleitung:** Ersetze jeden Platzhalter `[UNTERNEHMEN]` durch den Firmennamen (z. B. „Siemens AG"). Übergib den Prompt an ein LLM **mit Web-/Datenbank-Zugriff** oder nutze ihn als strukturierte Rechercheleitlinie.

---

## DER PROMPT

```
Du bist ein Senior Equity Research Analyst (CFA, 15+ Jahre Erfahrung in
fundamentaler Unternehmensbewertung). Deine Aufgabe:

  Recherchiere und schätze SÄMTLICHE Parameter für ein
  Sum-of-the-Parts (SOTP) Monte-Carlo-DCF-Modell
  für [UNTERNEHMEN].

Die Ergebnisse werden direkt in eine Streamlit-App eingegeben. Halte dich
EXAKT an die unten definierten Formate und Einheiten.


╔═══════════════════════════════════════════════════════════════════════╗
║  MODELLARCHITEKTUR – IMMER IM HINTERKOPF BEHALTEN                     ║
╚═══════════════════════════════════════════════════════════════════════╝

BEWERTUNGSANSATZ
  • FCFF-basierter DCF (Free Cash Flow to Firm) je Segment
  • Sum-of-the-Parts: Equity = Σ EV_i − PV(Holdingkosten)
      − Nettoverschuldung − Minderheiten − Pensionen
      + Nicht-operative Assets + Beteiligungen
  • Mid-Year Discounting Convention (FCFFs bei t − 0,5; TV bei T)

STOCHASTIK & SAMPLING
  • Monte-Carlo-Simulation (Standard: 50 000 Iterationen)
  • 3 Sampling-Strategien:
      ┌─────────────────────────┬──────────────────────────────────────┐
      │ Strategie               │ Eigenschaften                        │
      ├─────────────────────────┼──────────────────────────────────────┤
      │ Pseudo-Random (Standard)│ Klassisches MC, NumPy-RNG            │
      │ Antithetic Variates     │ u + (1−u) Spiegelung → Varianz↓     │
      │ Quasi-MC (Sobol)        │ Scrambled Sobol → schnellere Konverg.│
      └─────────────────────────┴──────────────────────────────────────┘

  Empfehlung: Sobol bei ≤ 7 stochastischen Parametern pro Segment
  (optimale Diskrepanz). Antithetic als guter Kompromiss. Pseudo-Random
  wenn maximale Flexibilität bei vielen korrelierten Segmenten nötig.

  • 6 Verteilungstypen pro Parameter:
      ┌────────────────────┬────────────────────────────────────────┐
      │ Typ                │ Eingabe-Parameter                      │
      ├────────────────────┼────────────────────────────────────────┤
      │ Fest               │ Wert                                   │
      │ Normalverteilung   │ μ, σ                                   │
      │ Lognormalverteilung│ μ, σ  (der unterliegenden Normalvert.) │
      │ Dreiecksverteilung │ Min, Mode, Max                         │
      │ Gleichverteilung   │ Min, Max                               │
      │ PERT-Verteilung    │ Min, Mode, Max  (λ = 4, intern)        │
      └────────────────────┴────────────────────────────────────────┘

  Entscheidungshilfe Verteilungswahl:
    → Hohe Sicherheit, historischer Fakt      → Fest
    → Symmetrische Unsicherheit               → Normal
    → Nur positive Werte, rechtsschief        → Lognormal
    → Min/Mode/Max bekannt, moderate Gewichte → PERT (BEVORZUGT)
    → Min/Mode/Max bekannt, Extremwerte fair  → Dreieck
    → Nur Grenzen bekannt, kein Mode          → Gleichverteilung

  ⚠️ PERT vor Dreieck bevorzugen (weniger Gewicht auf Extremwerte).
  ⚠️ Bei Unsicherheit: BREITERE Verteilung wählen – das ist der
     Mehrwert einer MC-Simulation gegenüber einem Punkt-DCF.

FADE-MODELL (Universal, exponentiell)
  Jeder Werttreiber kann von einem initialen zu einem terminalen Wert
  konvergieren:

    p(t) = p_terminal + (p_initial − p_terminal) · e^(−λ · t)

  • p_initial = gesampelter Startwert (aus der Verteilung)
  • p_terminal = langfristiges Gleichgewicht (eigene Verteilung)
  • λ = Fade-Geschwindigkeit (ein λ pro Segment, gilt für ALLE
    Parameter dieses Segments)
  • Fade unterstützt für: Umsatzwachstum, EBITDA-Marge, D&A,
    Steuersatz, CAPEX, NWC

CROSS-SEGMENT-KORRELATION
  Gauss-Copula: Ein Korrelationskoeffizient ρ pro Segmentpaar
  korreliert ALLE stochastischen Parameter beider Segmente
  gleichzeitig, ohne die Marginalverteilungen zu verzerren.

INTRA-SEGMENT-COPULA (NEU)
  7×7-Gauss-Copula INNERHALB jedes Segments für die 7 Kern-Werttreiber:
    Revenue Growth, EBITDA-Marge, D&A, Steuersatz, CAPEX, NWC, WACC

  Modelliert ökonomische Abhängigkeiten wie:
    • Wachstum ↔ Marge (Skaleneffekte: ρ > 0)
    • Marge ↔ CAPEX (Investitionsintensität: ρ < 0 oder ρ > 0)
    • WACC ↔ Wachstum (risikoreichere Segmente wachsen schneller)

  ⚠️ Dies ist OPTIONAL. Nur aktivieren, wenn fundierte Schätzungen
     für die paarweisen Abhängigkeiten vorliegen.

QUALITÄTSMETRIKEN (automatisch berechnet, aber relevant für Checks)
  • TV/EV-Ratio pro Segment (PV des Terminal Value / Enterprise Value)
  • Implied ROIC pro Segment
  • Reinvestment Rate pro Segment
  • SOTP-Treemap (proportionale Segment-EV-Visualisierung)
  • Composite Quality Score (0–100):
      TV/EV-Sub (0–25) + Konvergenz-Sub (0–25)
      + Sensitivity-Sub (0–25) + Dispersion-Sub (0–25)

EINHEITEN
  • Alle Prozentangaben als %, NICHT als Dezimalzahl.
    Beispiel: 5,0 % Wachstum, NICHT 0,05. Die App konvertiert intern.
  • Berichtswährung von [UNTERNEHMEN] konsistent nutzen und EINMALIG
    am Anfang nennen.


═══════════════════════════════════════════════════════════════════════════
SCHRITT 1 · SEGMENTIDENTIFIKATION
═══════════════════════════════════════════════════════════════════════════

Analysiere die Geschäftsstruktur von [UNTERNEHMEN]:

1. Identifiziere alle wesentlichen Geschäftssegmente anhand der
   aktuellsten Segmentberichterstattung (Annual Report / 10-K / 20-F).
2. Für jedes Segment ermittle:
   a) Segmentname (exakt wie im Geschäftsbericht)
   b) Kurzbeschreibung (1–2 Sätze: Was macht das Segment?)
   c) Letzter berichteter Jahresumsatz (Mio. Berichtswährung)
   d) Strategische Positionierung:
      Wachstumsmotor | Cash Cow | Turnaround | Restrukturierung
   e) Sektorklassifikation (für Portfolio-App-Brücke):
      Technologie | Konsumgüter | Gesundheit | Finanzen | Energie |
      Industrie | Grundstoffe | Immobilien | Telekommunikation |
      Versorger | Sonstige
3. Sortiere absteigend nach Umsatz.

Ausgabe als Tabelle:

| # | Segment | Umsatz (Mio.) | Beschreibung | Positionierung | Sektor |
|---|---------|---------------|--------------|----------------|--------|
| 1 | …       | …             | …            | …              | …      |


═══════════════════════════════════════════════════════════════════════════
SCHRITT 2 · FCFF-PARAMETER JE SEGMENT (11 Parameter + Fade-Terminals)
═══════════════════════════════════════════════════════════════════════════

Für JEDES Segment in Schritt 1: Recherchiere und schätze die folgenden
Parameter. Liefere pro Parameter:

  (a) Punktschätzung (Base Case)
  (b) Empfohlene Verteilung mit allen Parametern
  (c) Terminal-Wert (langfristiger Zielwert, falls Fade empfohlen)
  (d) Begründung (2–4 Sätze) mit konkreten Quellen

Falls ein Parameter nicht segmentspezifisch berichtet wird → Konzernwert
verwenden und mit „≈ Konzernannahme" kennzeichnen.

────────────────────────────────────────────────────────────────────────
2.1  Basisumsatz (Mio.)
────────────────────────────────────────────────────────────────────────
App-Feld: „Basisumsatz (Mio. / Jahr 0)"
• Letzter tatsächlich berichteter Segmentumsatz des jüngsten GJ.
• IMMER Fest (historischer Fakt).
• Quelle: Segmentberichterstattung.

────────────────────────────────────────────────────────────────────────
2.2  Prognosezeitraum (Jahre)
────────────────────────────────────────────────────────────────────────
App-Feld: „Detail-Prognosezeitraum (Jahre)"  [1–30, ganzzahlig]
• Reife/stabile Segmente → 5 Jahre
• Wachstumssegmente → 7–10 Jahre (bis Steady-State plausibel)
• Zyklische Segmente → voller Konjunkturzyklus (5–7 Jahre)
• IMMER Fest.

────────────────────────────────────────────────────────────────────────
2.3  Umsatzwachstum (%)
────────────────────────────────────────────────────────────────────────
App-Feld: „Umsatzwachstum" (Verteilungsinput)
Wachstumsmodell: „Konstant" oder „Fade-Modell"

Recherchiere:
  • Historisches CAGR (3 J / 5 J) des Segments
  • Konsensus-Analystenprognosen (Bloomberg, LSEG, Visible Alpha)
  • Branchenwachstum / TAM (Gartner, IDC, Statista)
  • Unternehmensguidance

Verteilung: PERT bevorzugt (Min/Mode/Max). Normal bei Symmetrie.

Fade (dringend empfohlen):
  Wachstums-Modell → „Fade-Modell" in der App wählen.
  → g_initial = die hier angegebene Verteilung
  → g_terminal = langfristiges nominales BIP-Wachstum (ca. 2–3 %)
  → Die App lässt g über den Prognosezeitraum exponentiell
    von g_initial nach g_terminal konvergieren.
  Nenne explizit: g_initial (Verteilung) UND g_terminal (%).

────────────────────────────────────────────────────────────────────────
2.4  EBITDA-Marge (%)
────────────────────────────────────────────────────────────────────────
App-Feld: „EBITDA-Marge" (Verteilungsinput)

Recherchiere:
  • Historische Segmentmarge (3–5 J, Trend?)
  • Peer-Group Median (Pure-Play-Vergleiche)
  • Skaleneffekte & operative Leverage
  • Management-Zielvorgaben („Mid-term targets")

Verteilung: PERT bevorzugt. Normal bei Symmetrie.

Fade-Terminal (empfohlen):
  → Langfristig nachhaltige EBITDA-Marge.
  → Marge über Peer-Median? → Fade nach unten (Mean Reversion).
  → Wachstumssegment mit Skalierung? → Fade nach oben.

────────────────────────────────────────────────────────────────────────
2.5  D&A als % vom Umsatz
────────────────────────────────────────────────────────────────────────
App-Feld: „D&A (% Umsatz)" (Verteilungsinput)

Recherchiere:
  • Historische D&A/Umsatz-Ratio (Segment oder Konzern)
  • Kapitalintensität der Branche
  • Investitionszyklen (neue Assets → höhere D&A)

Verteilung: Fest oder Dreieck/PERT (D&A ist relativ stabil).

Fade-Terminal (optional):
  → Bei reifem Asset-Bestand sinkt D&A/Umsatz langfristig zum
    Maintenance-Level.

────────────────────────────────────────────────────────────────────────
2.6  Effektiver Steuersatz (%)
────────────────────────────────────────────────────────────────────────
App-Feld: „Steuersatz" (Verteilungsinput)

Recherchiere:
  • Historischer effektiver Konzern-ETR (3 J)
  • Körperschaftssteuersatz des Sitzlandes
  • Sondereffekte (Verlustvorträge, Pillar-Two 15 %)
  • Langfristig nachhaltiger ETR

Verteilung: Fest (typisch). Dreieck bei Steuerreform-Unsicherheit.

Fade-Terminal (optional):
  → Nur bei erwarteten regulatorischen Änderungen.

────────────────────────────────────────────────────────────────────────
2.7  CAPEX als % vom Umsatz
────────────────────────────────────────────────────────────────────────
App-Feld: „CAPEX (% Umsatz)" (Verteilungsinput)

Recherchiere:
  • Historische CAPEX/Umsatz-Quote (Trend?)
  • Management-Guidance zu Investitionen
  • Branchenvergleich:
      Software ~3–5 % | Telko ~15–20 % | Industrie ~5–8 %
  • Maintenance vs. Growth CAPEX (falls berichtet)

Verteilung: PERT oder Dreieck.

Fade-Terminal (dringend empfohlen):
  → Terminal-CAPEX = Erhaltungsinvestitionen (Maintenance CAPEX).
  → Faustregel: Maintenance CAPEX ≈ D&A langfristig.
  → Growth CAPEX entfällt im Steady State.

────────────────────────────────────────────────────────────────────────
2.8  NWC-Veränderung als % der Umsatzveränderung
────────────────────────────────────────────────────────────────────────
App-Feld: „NWC (% ΔUmsatz)" (Verteilungsinput)

Recherchiere:
  • Historische NWC/Umsatz-Quote und deren Veränderung
  • Branchencharakteristik:
      Software/Digital 0–5 % | Handel/Industrie 10–20 % | Bau 15–25 %
  • Cash Conversion Cycle des Segments

Verteilung: Fest oder PERT.

Fade-Terminal (optional):
  → Nur wenn Produktmix-Shift die NWC-Intensität verändert.

────────────────────────────────────────────────────────────────────────
2.9  WACC (%)
────────────────────────────────────────────────────────────────────────
App-Feld: „WACC" (Verteilungsinput)

Berechne segmentspezifisch und ZEIGE die vollständige Herleitung:

  a) Eigenkapitalkosten k_e (CAPM):
     • Risikofreier Zins: 10Y-Staatsanleihe des Referenzmarktes
     • ERP: Damodaran Equity Risk Premium für relevanten Markt
     • Beta: 3–5 Pure-Play-Peers → Levered Beta → Unlever (Hamada)
       → Re-lever mit Kapitalstruktur von [UNTERNEHMEN]
     • Ggf. Size Premium / Country Risk Premium

  b) Fremdkapitalkosten k_d:
     • Credit Rating → Credit Spread
     • Oder: Ø Zinsaufwand / Finanzschulden (historisch)

  c) Kapitalstruktur (Marktwertbasis):
     • E/V und D/V
     • Zielkapitalstruktur falls kommuniziert

  d) WACC = (E/V · k_e) + (D/V · k_d · (1 − t))

Verteilung: Normal oder PERT (σ typisch 0,5–2,0 Prozentpunkte).

Hinweis: Die App erzwingt WACC ≥ 0,5 % als Sicherheitsgrenze.

────────────────────────────────────────────────────────────────────────
2.10  Terminal-Value-Methode & Parameter
────────────────────────────────────────────────────────────────────────
App-Feld: „Terminal-Value-Methode" (Selectbox)

Empfehle PRO SEGMENT eine der beiden Methoden und begründe:

OPTION A – Gordon Growth Model
  App-Feld: „TV-Wachstumsrate" (Verteilungsinput)
  • g = langfristiges nachhaltiges Wachstum
  • Benchmark: nominales BIP-Wachstum (2–3 %)
  • MUSS deutlich < WACC sein (Faustregel: g < WACC − 3 pp)
  • Schrumpfende Segmente: 0 % oder negativ
  • ALS VERTEILUNG angeben! PERT bevorzugt.

OPTION B – Exit Multiple
  App-Feld: „Exit-Multiple (EV/EBITDA)" (Verteilungsinput)
  • Basierend auf: aktuellen Peer-Trading-Multiples,
    historischen M&A-Transaktionsmultiples,
    langfristigem Branchendurchschnitt
  • ALS VERTEILUNG angeben! PERT oder Dreieck.

  ⚠️ SONDERFALL: Exit Multiple + Fade-Modell
    Wenn für ein Segment das Fade-Modell UND Exit Multiple gewählt
    werden, muss zusätzlich ein „Langfristiges Umsatzwachstum
    (Fade-Ziel)" angegeben werden:
    App-Feld: „Langfristiges Umsatzwachstum (Fade-Ziel)"
    → Typisch: nominales BIP-Wachstum (2–3 %). PERT empfohlen.

Leitlinie:
  • Stabile, vorhersagbare FCFF → Gordon Growth
  • Zyklisch / M&A-aktiv / Branchenpremium → Exit Multiple
  • Tech mit Margentestung → Gordon Growth

────────────────────────────────────────────────────────────────────────
2.11  Fade-Geschwindigkeit λ
────────────────────────────────────────────────────────────────────────
App-Feld: „Fade-Geschwindigkeit (λ)" (Slider, 0,05–3,0)

Ein einziger λ-Wert pro Segment, der für ALLE Parameter dieses
Segments gilt (Umsatzwachstum, EBITDA-Marge, D&A, Steuer, CAPEX, NWC).

Orientierung:
  • 0,2–0,3 = langsam (langfristige strukturelle Shifts)
  • 0,5     = mittel  (Standard-Empfehlung)
  • 0,8–1,5 = schnell (kurzfristige Normalisierung)
  • > 1,5   = sehr schnell (fast sofortige Normalisierung)

FEST. Begründe die Wahl.

────────────────────────────────────────────────────────────────────────
2.12  Parameter-Fade: Terminal-Werte (optional, aber empfohlen)
────────────────────────────────────────────────────────────────────────
App: Checkbox „Parameter-Fade aktivieren" pro Segment.
Wenn aktiviert, erscheinen Terminal-Verteilungsinputs für:

  • EBITDA-Marge (Terminal)     → langfristig nachhaltige Marge
  • D&A (Terminal)              → Maintenance-Level
  • Steuersatz (Terminal)       → nachhaltiger ETR
  • CAPEX (Terminal)            → Maintenance CAPEX
  • NWC (Terminal)              → langfristige NWC-Intensität

Jeder Terminal-Wert ist eine eigene Verteilung (gleiche 6 Typen).

Empfehlung: Mindestens EBITDA-Marge und CAPEX mit Terminal-Werten
versehen. Diese haben den größten Einfluss auf den Fair Value.


═══════════════════════════════════════════════════════════════════════════
SCHRITT 3 · ERWEITERTE EQUITY BRIDGE (8 Posten)
═══════════════════════════════════════════════════════════════════════════

Alle 8 Posten werden in der App als Verteilungsinput eingegeben
(Standard: Fest). Für jeden Posten: Nenne den Wert UND entscheide,
ob eine Verteilung angebracht ist.

Verteilung empfohlen bei: erwartetem Schuldenabbau, laufenden
Aktienrückkäufen, unsicherer Pensionsbewertung, Marktwert ≠ Buchwert
bei Beteiligungen.

────────────────────────────────────────────────────────────────────────
3.1  Holdingkosten (Mio. p.a.)                         → Abzug (PV)
────────────────────────────────────────────────────────────────────────
App-Feld: „Holdingkosten (Mio. p.a.)" (Verteilungsinput)
• Nicht segmentzugeordnete Kosten aus der Segmentüberleitung
• Inkl. Vorstandsvergütung, zentrale IT, Konzernfunktionen
• Quelle: Segmentüberleitung im Geschäftsbericht
• App berechnet PV als Perpetuity: PV = Holdingkosten / Diskontierungssatz

────────────────────────────────────────────────────────────────────────
3.2  Diskontierungssatz (%)                             → für PV(Holding)
────────────────────────────────────────────────────────────────────────
App-Feld: „Diskontierungssatz (%)" (Verteilungsinput)
• Typisch: Konzern-WACC
• Auch als Verteilung modellierbar

────────────────────────────────────────────────────────────────────────
3.3  Nettoverschuldung (Mio.)                           → Abzug
────────────────────────────────────────────────────────────────────────
App-Feld: „Nettoverschuldung (Mio.)" (Verteilungsinput)
• Net Debt = Finanzschulden − Cash & Äquivalente − Finanzanlagen
• KEINE Pensionen oder Leasing hier (→ separate Posten)
• Quelle: Letzte berichtete Bilanz

────────────────────────────────────────────────────────────────────────
3.4  Aktien ausstehend (Mio., voll verwässert)          → Divisor
────────────────────────────────────────────────────────────────────────
App-Feld: „Aktien ausstehend (Mio.)" (Verteilungsinput)
• Basic Shares + Verwässerung (Treasury Stock Method)
• Verteilung bei laufenden Rückkaufprogrammen

────────────────────────────────────────────────────────────────────────
3.5  Minderheitsanteile (Mio.)                          → Abzug
────────────────────────────────────────────────────────────────────────
App-Feld: „Minderheitsanteile (Mio.)"
• Buchwert Non-controlling Interests
• Falls nicht vorhanden → Fest: 0

────────────────────────────────────────────────────────────────────────
3.6  Pensionsrückstellungen (Mio.)                      → Abzug
────────────────────────────────────────────────────────────────────────
App-Feld: „Pensionsrückstellungen (Mio.)"
• Netto-Pensionsverpflichtung = DBO − Plan Assets
• Falls nicht vorhanden → Fest: 0

────────────────────────────────────────────────────────────────────────
3.7  Nicht-operative Assets (Mio.)                      → Zuschlag
────────────────────────────────────────────────────────────────────────
App-Feld: „Nicht-operative Assets (Mio.)"
• Equity-Method-Beteiligungen, überschüssige Immobilien,
  langfristige Finanzanlagen außerhalb des operativen Kerns
• Falls nicht vorhanden → Fest: 0

────────────────────────────────────────────────────────────────────────
3.8  Beteiligungen (Mio.)                               → Zuschlag
────────────────────────────────────────────────────────────────────────
App-Feld: „Beteiligungen (Mio.)"
• Equity-Buchwert strategischer Minderheitsbeteiligungen
• Falls nicht vorhanden → Fest: 0


═══════════════════════════════════════════════════════════════════════════
SCHRITT 4 · CROSS-SEGMENT-KORRELATION
═══════════════════════════════════════════════════════════════════════════

Bei ≥ 2 Segmenten: Schätze für JEDES Segmentpaar (i, j) mit i < j
einen Korrelationskoeffizienten ρ_ij ∈ [−1, 1].

Orientierungshilfe:

| Beziehung                                    | Typisches ρ  |
|----------------------------------------------|-------------|
| Gleiche Branche, gleiche Region              | 0,6 – 0,9  |
| Gleiche Branche, andere Region               | 0,3 – 0,6  |
| Komplementär (z. B. Hardware + Services)     | 0,2 – 0,5  |
| Diversifiziert (z. B. Industrie + Finanz)    | 0,0 – 0,3  |
| Gegenläufig (z. B. Upstream + Downstream)    | −0,2 – 0,1 |

Hinweis: Die App validiert automatisch positiv-semidefinite Korrelation.


═══════════════════════════════════════════════════════════════════════════
SCHRITT 4b · INTRA-SEGMENT-COPULA (OPTIONAL)
═══════════════════════════════════════════════════════════════════════════

Die App unterstützt optional eine 7×7-Gauss-Copula INNERHALB jedes
Segments, die Abhängigkeiten zwischen den 7 Werttreibern modelliert:

  Revenue Growth, EBITDA-Marge, D&A, Steuersatz, CAPEX, NWC, WACC

Falls aktiviert, schätze für jedes Segment die paarweisen
Korrelationskoeffizienten zwischen diesen Parametern.

Typische Muster:

| Paar                    | Richtung | Typisches ρ | Begründung |
|-------------------------|----------|-------------|------------|
| Wachstum ↔ Marge        | +        | 0,1 – 0,4  | Skaleneffekte |
| Wachstum ↔ CAPEX        | +        | 0,2 – 0,5  | Investition für Wachstum |
| Marge ↔ NWC             | −        | −0,3 – 0,0 | Höhere Marge → strafferes WC |
| WACC ↔ Wachstum         | +        | 0,1 – 0,3  | Risiko-Rendite-Tradeoff |
| D&A ↔ CAPEX             | +        | 0,3 – 0,6  | Investition → Abschreibung |
| Steuer ↔ andere         | ≈ 0      | 0,0 – 0,1  | Steuer weitgehend extern |

⚠️ NUR aktivieren, wenn fundierte Schätzungen möglich. Bei Unsicherheit
   besser weglassen – die App funktioniert auch ohne Intra-Segment-Copula.

Ausgabeformat: 7×7-Matrix pro Segment (nur untere Dreieck nötig).


═══════════════════════════════════════════════════════════════════════════
SCHRITT 5 · SIMULATIONSPARAMETER
═══════════════════════════════════════════════════════════════════════════

| Parameter               | Empfehlung       | App-Feld             |
|-------------------------|------------------|----------------------|
| MC-Iterationen          | 50 000           | Anzahl MC-Iterationen|
| Random Seed             | 42               | Random Seed          |
| Mid-Year Convention     | Ja (Standard)    | Mid-Year Discounting |
| Sampling-Methode        | Sobol (oder PR)  | Sampling-Methode     |

Sampling-Empfehlung:
  • 1–3 Segmente, wenig stochastische Params → Quasi-MC (Sobol)
  • Viele Segmente + Cross-Segment-Copula → Pseudo-Random
  • Varianzreduktion ohne Sobol-Limitierungen → Antithetic Variates


═══════════════════════════════════════════════════════════════════════════
SCHRITT 6 · AUSGABETABELLEN (Direkt in die App übertragbar)
═══════════════════════════════════════════════════════════════════════════

Liefere die Ergebnisse in EXAKT diesen Tabellen:

─────────────── FÜR JEDES SEGMENT EINE TABELLE ───────────────

### SEGMENT [N]: [Name]

| Parameter | Wachstumsmodell | Verteilung | Min | Mode | Max | μ | σ | Terminal | λ | Begründung |
|---|---|---|---|---|---|---|---|---|---|---|
| Basisumsatz (Mio.) | – | FEST: [Wert] | – | – | – | – | – | – | – | [Quelle, GJ] |
| Prognosejahre | – | FEST: [N] | – | – | – | – | – | – | – | [Warum N Jahre?] |
| Umsatzwachstum (%) | [Konstant/Fade] | [Typ]: … | … | … | … | … | … | [g_terminal] | [λ] | [Hist. CAGR, Konsensus, TAM, Guidance] |
| EBITDA-Marge (%) | – | [Typ]: … | … | … | … | … | … | [m_terminal] | – | [Hist., Peer-Median, Skaleneffekte] |
| D&A (% Umsatz) | – | [Typ]: … | … | … | … | … | … | [d_terminal] | – | [Hist. Ratio, Kapitalintensität] |
| Steuersatz (%) | – | [Typ]: … | … | … | … | … | … | [t_terminal] | – | [ETR-Analyse, Jurisdiktion] |
| CAPEX (% Umsatz) | – | [Typ]: … | … | … | … | … | … | [c_terminal] | – | [Maintenance vs. Growth, Guidance] |
| NWC (% ΔUmsatz) | – | [Typ]: … | … | … | … | … | … | [n_terminal] | – | [CCC, Branchenvergleich] |
| WACC (%) | – | [Typ]: … | … | … | … | … | … | – | – | [CAPM komplett zeigen!] |
| TV-Methode | – | [Gordon Growth / Exit Multiple] | – | – | – | – | – | – | – | [Warum diese Methode?] |
| TV-Wachstum (%) | – | [Typ]: … | … | … | … | … | … | – | – | [BIP-Benchmark, WACC-Spread] ★ |
| ODER: Exit-Multiple | – | [Typ]: … | … | … | … | … | … | – | – | [Peer-Multiples, M&A-Comps] ★ |

★ ALS VERTEILUNG, nicht Fest. PERT bevorzugt.

Hinweise:
  • „Terminal": Langfristiger Zielwert für Parameter-Fade. „–" wenn kein Fade.
  • „λ": Nur in Zeile Umsatzwachstum; gilt für ALLE Fade-Parameter.
  • Nicht benötigte Spalten mit „–" füllen.

─────────────── CORPORATE BRIDGE ───────────────

### CORPORATE BRIDGE

| Posten | Wert / Verteilung | Wirkung | Begründung |
|---|---|---|---|
| Holdingkosten p.a. (Mio.) | [Wert oder Verteilung] | Abzug (PV) | [Quelle] |
| Diskontierungssatz (%) | [Wert oder Verteilung] | für PV(Holding) | [= Konzern-WACC] |
| Nettoverschuldung (Mio.) | [Wert oder Verteilung] | Abzug | [Berechnung zeigen] |
| Aktien (Mio., verwässert) | [Wert oder Verteilung] | Divisor | [Quelle] |
| Minderheitsanteile (Mio.) | [Wert oder 0] | Abzug | [Quelle] |
| Pensionsrückstellungen (Mio.) | [Wert oder 0] | Abzug | [DBO − Plan Assets] |
| Nicht-operative Assets (Mio.) | [Wert oder 0] | Zuschlag | [Welche?] |
| Beteiligungen (Mio.) | [Wert oder 0] | Zuschlag | [Welche?] |

─────────────── KORRELATIONSMATRIX ───────────────

### CROSS-SEGMENT-KORRELATION

| Segment i | Segment j | ρ | Begründung |
|---|---|---|---|
| … | … | … | [Warum diese Höhe?] |

─────────────── INTRA-SEGMENT-COPULA (optional) ───────────────

### INTRA-SEGMENT-COPULA [Segmentname]

Nur ausfüllen, wenn explizit empfohlen.

| Param 1 | Param 2 | ρ | Begründung |
|---|---|---|---|
| Revenue Growth | EBITDA-Marge | … | … |
| … | … | … | … |

─────────────── SIMULATIONSPARAMETER ───────────────

### SIMULATIONSPARAMETER

| Parameter | Wert |
|---|---|
| MC-Iterationen | 50 000 |
| Random Seed | 42 |
| Mid-Year Convention | Ja |
| Sampling-Methode | [Pseudo-Random / Antithetic / Sobol] |


═══════════════════════════════════════════════════════════════════════════
SCHRITT 7 · PLAUSIBILITÄTS- & QUALITÄTSCHECKS (9 Checks)
═══════════════════════════════════════════════════════════════════════════

Führe ALLE 9 Checks durch. Dokumentiere das Ergebnis explizit:
  ✅ bestanden | ⚠️ Anpassung nötig | ❌ Problem

Bei ⚠️ oder ❌: Schätzungen in Schritten 2–3 ANPASSEN und die
korrigierte Begründung ergänzen.

────────────────────────────────────────────────────────────────────────
Check 1 · Umsatz-Cross-Check
────────────────────────────────────────────────────────────────────────
Σ Basisumsätze aller Segmente ≈ berichteter Konzernumsatz?
Abweichung < 5 % → ✅. Sonst: Differenz erklären.

────────────────────────────────────────────────────────────────────────
Check 2 · Margin-Cross-Check
────────────────────────────────────────────────────────────────────────
Umsatzgewichteter Durchschnitt der Segment-EBITDA-Margen ≈
berichtete Konzern-EBITDA-Marge?

────────────────────────────────────────────────────────────────────────
Check 3 · WACC-Plausibilität
────────────────────────────────────────────────────────────────────────
Liegt jeder Segment-WACC in einem plausiblen Bereich für Branche,
Bonität und aktuelles Zinsumfeld?

────────────────────────────────────────────────────────────────────────
Check 4 · TV-Growth ≪ WACC
────────────────────────────────────────────────────────────────────────
Für JEDES Gordon-Growth-Segment: g < WACC − 3 pp?
Falls nicht → g senken.

────────────────────────────────────────────────────────────────────────
Check 5 · TV/EV-Ratio (Vorab-Schätzung)
────────────────────────────────────────────────────────────────────────
Schätze grob PV(TV) / EV je Segment:
  < 60 %  → ✅  |  60–75 % → ⚠️  |  > 75 % → ❌

────────────────────────────────────────────────────────────────────────
Check 6 · Implied ROIC
────────────────────────────────────────────────────────────────────────
ROIC ≈ NOPAT-Marge × (1 + g) / Reinvestment Rate
Plausibel für die Branche?
  Software 25–50 % | Industrie 10–20 % | Handel 8–15 %

────────────────────────────────────────────────────────────────────────
Check 7 · Fade-Konsistenz
────────────────────────────────────────────────────────────────────────
Konvergieren alle Terminal-Werte zu branchenüblichen Niveaus?
Ist die Fade-Richtung ökonomisch sinnvoll?

────────────────────────────────────────────────────────────────────────
Check 8 · Korrelationslogik
────────────────────────────────────────────────────────────────────────
Gleicher Endmarkt → höhere Korrelation. Matrix konsistent?
Intra-Segment-Copula: Vorzeichenlogik prüfen (Wachstum↔CAPEX > 0?).

────────────────────────────────────────────────────────────────────────
Check 9 · Impliziter Fair Value (Überschlagsrechnung)
────────────────────────────────────────────────────────────────────────
Equity ≈ (Σ EV − PV(Holding) − NetDebt − Minority − Pension
          + NonOp + Associates) / Aktien
Vergleiche mit aktuellem Aktienkurs: Premium/Discount in %.
> 100 % Abweichung → Parameterfehler wahrscheinlich.

────────────────────────────────────────────────────────────────────────

Ergebnistabelle:

| # | Check | Ergebnis | Detail / Kommentar |
|---|-------|----------|--------------------|
| 1 | Umsatz-Cross-Check | ✅/⚠️/❌ | … |
| 2 | Margin-Cross-Check | ✅/⚠️/❌ | … |
| 3 | WACC-Plausibilität | ✅/⚠️/❌ | … |
| 4 | TV-Growth ≪ WACC | ✅/⚠️/❌ | … |
| 5 | TV/EV-Ratio | ✅/⚠️/❌ | … |
| 6 | Implied ROIC | ✅/⚠️/❌ | … |
| 7 | Fade-Konsistenz | ✅/⚠️/❌ | … |
| 8 | Korrelationslogik | ✅/⚠️/❌ | … |
| 9 | Impliziter Fair Value | ✅/⚠️/❌ | … |


═══════════════════════════════════════════════════════════════════════════
SCHRITT 8 · QUELLEN & DATENBASIS
═══════════════════════════════════════════════════════════════════════════

| Typ | Quelle | Datum / GJ |
|---|---|---|
| Geschäftsbericht | [UNTERNEHMEN] Annual Report [GJ] | … |
| Analystenberichte | [Bloomberg Consensus / LSEG / …] | … |
| Branchenreports | [Gartner / IDC / McKinsey / Statista / …] | … |
| Damodaran | [Betas, ERP, Multiples – pages.stern.nyu.edu/~adamodar/] | … |
| Marktdaten | [Anleiherenditen, CDS, Ratings] | … |
| Sonstige | … | … |


═══════════════════════════════════════════════════════════════════════════
ZUSAMMENFASSUNG (EXECUTIVE SUMMARY)
═══════════════════════════════════════════════════════════════════════════

Schließe mit einer Executive Summary (max. 5 Sätze):
  1. Berichtswährung + Anzahl Segmente
  2. Wichtigste Werttreiber-Unsicherheiten (die breitesten Verteilungen)
  3. Welche Segmente treiben den Wert, welche sind marginal?
  4. Erwartete Bewertungsrichtung vs. aktueller Kurs
  5. Empfehlung zur Interpretation der MC-Ergebnisse
     (auf welche Quantile achten, wo liegt das Risiko?)
```

---

## Beispiel-Anwendung

Ersetze `[UNTERNEHMEN]` und führe den Prompt aus:

```
[UNTERNEHMEN] = Siemens AG
```
