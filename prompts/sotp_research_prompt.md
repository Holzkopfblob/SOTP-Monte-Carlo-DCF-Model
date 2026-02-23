# SOTP Monte-Carlo DCF – Research & Estimation Prompt

> **Verwendung:** Kopiere diesen Prompt und ersetze jeden Platzhalter `[UNTERNEHMEN]` durch den tatsächlichen Firmennamen (z. B. „Siemens AG", „Alphabet Inc.", „SAP SE"). Übergib den Prompt an ein LLM **mit Webzugriff** oder nutze ihn als strukturierte Rechercheleitlinie.

---

## DER PROMPT

```
Du bist ein „Senior Equity Research Analyst" mit CFA-Qualifikation und
15 Jahren Erfahrung in fundamentaler Unternehmensbewertung.

Deine Aufgabe: Recherchiere und schätze **ALLE** Parameter, die für ein
Sum-of-the-Parts (SOTP) Monte-Carlo-DCF-Modell für **[UNTERNEHMEN]**
benötigt werden.

Das Modell hat folgende Eigenschaften – behalte sie bei jeder Schätzung
im Hinterkopf:

• FCFF-Ansatz (Free Cash Flow to Firm)
• 6 Verteilungstypen:
    Fest | Normal (μ, σ) | Lognormal (μ, σ) | Dreieck (Min, Mode, Max) |
    Gleichverteilung (Min, Max) | PERT (Min, Mode, Max)
• Universal-Fade-Modell – jeder Werttreiber (Wachstum, Marge, D&A, Steuer,
  CAPEX, NWC) kann exponentiell von einem initialen zu einem terminalen Wert
  konvergieren:  p_t = p_terminal + (p_initial − p_terminal) · e^(−λ·t)
• Gauss-Copula für Cross-Segment-Korrelation
• Erweiterte Equity Bridge (Holding, Net Debt, Aktien, Minderheiten,
  Pensionen, nicht-operative Assets, Beteiligungen) – jeder Posten
  optional als Verteilung
• Bewertungsqualitäts-Metriken: TV/EV-Ratio, Implied ROIC, Reinvestment
  Rate, Composite Quality Score (0–100)
• Mid-Year Discounting Convention (Standard)

Alle Prozentangaben als %, NICHT als Dezimalzahl (das Modell konvertiert
intern). Berichtswährung konsistent nutzen und einmalig nennen.

Bei Unsicherheit: Lieber eine BREITERE Verteilung wählen – das ist der
Mehrwert einer MC-Simulation gegenüber einem Punkt-DCF.
PERT wird gegenüber Dreieck bevorzugt (weniger Gewicht auf Extremwerte).


═══════════════════════════════════════════════════════════════════════════
SCHRITT 1 · SEGMENTIDENTIFIKATION
═══════════════════════════════════════════════════════════════════════════

Analysiere die Geschäftsstruktur von [UNTERNEHMEN]:

1. Identifiziere alle wesentlichen Geschäftssegmente anhand der letzten
   Segmentberichterstattung (Annual Report / 10-K / 20-F).
2. Für jedes Segment:
   a) Segmentname (wie im Geschäftsbericht berichtet)
   b) Kurzbeschreibung (1–2 Sätze: Was macht das Segment? Kernprodukte/Märkte?)
   c) Letzter berichteter Jahresumsatz (Mio. Berichtswährung)
   d) Strategische Positionierung: Wachstumsmotor / Cash Cow / Turnaround / Restrukturierung
3. Sortiere absteigend nach Umsatz.

Liefere als Tabelle:

| # | Segment | Umsatz (Mio.) | Beschreibung | Positionierung |
|---|---------|---------------|--------------|----------------|
| 1 | …       | …             | …            | …              |


═══════════════════════════════════════════════════════════════════════════
SCHRITT 2 · FCFF-WERTTREIBER JE SEGMENT
═══════════════════════════════════════════════════════════════════════════

Für **jedes** Segment in Schritt 1 recherchiere die folgenden 11 Parameter.
Liefere für JEDEN Parameter:

  (a) **Punktschätzung** (Base Case)
  (b) **Empfohlene Verteilung** mit sämtlichen Parametern
  (c) **Terminal-Wert** (langfristiges Gleichgewicht, falls Fade empfohlen)
  (d) **Begründung** (2–4 Sätze) mit konkreten Quellenhinweisen

Wenn ein Parameter für ein Segment NICHT separat berichtet wird, nutze den
Konzernwert und kennzeichne dies als „Konzernannahme".

────────────────────────────────────────────────────────────────────────
2.1  Basisumsatz (Mio.) – Jahr 0
────────────────────────────────────────────────────────────────────────
• Letzter tatsächlich berichteter Jahresumsatz des Segments.
• Quelle: Segmentberichterstattung im letzten Geschäftsbericht.
• IMMER als FEST (historischer Fakt, keine Verteilung).

────────────────────────────────────────────────────────────────────────
2.2  Prognosezeitraum (Jahre)
────────────────────────────────────────────────────────────────────────
Empfehlungen:
  - Stabile/reife Segmente → 5 Jahre
  - Wachstumssegmente     → 7–10 Jahre (bis Steady-State erreicht)
  - Zyklische Segmente    → voller Konjunkturzyklus (5–7 Jahre)
• IMMER als FEST (ganzzahlig, 1–30).

────────────────────────────────────────────────────────────────────────
2.3  Umsatzwachstum (%)
────────────────────────────────────────────────────────────────────────
Recherchiere:
  • Historisches CAGR (3 J / 5 J) des Segments
  • Konsensus-Analystenprognosen (Bloomberg, LSEG, Visible Alpha)
  • Branchenwachstum (TAM/SAM via Gartner, IDC, Statista etc.)
  • Unternehmensguidance

Verteilung: PERT oder Dreieck (Min/Mode/Max).
             Normal bei symmetrischer Unsicherheit.

Fade-Terminal-Wert (dringend empfohlen):
  → Die Wachstumsrate sollte zur langfristigen Rate konvergieren
    (oft nahe nominalem BIP-Wachstum 2–3 %).
  → Nenne explizit: g_initial (%) und g_terminal (%).
  → Das Modell berechnet den Pfad automatisch via λ-Fade.

────────────────────────────────────────────────────────────────────────
2.4  EBITDA-Marge (%)
────────────────────────────────────────────────────────────────────────
Recherchiere:
  • Historische Segmentmarge (3–5 J, Trend?)
  • Peer-Group Median-Marge (Pure-Play-Vergleiche)
  • Skaleneffekte / operative Leverage
  • Management-Zielvorgaben („Mid-term targets")

Verteilung: PERT bevorzugt, Normal bei symmetrischer Unsicherheit.

Fade-Terminal-Wert (empfohlen):
  → Terminal-EBITDA-Marge = langfristig nachhaltige Marge.
  → Liegt die aktuelle Marge über Peer-Median? → Fade nach unten.
  → Wachstumssegment mit Margensteigerungspotenzial? → Fade nach oben.

────────────────────────────────────────────────────────────────────────
2.5  D&A als % vom Umsatz
────────────────────────────────────────────────────────────────────────
Recherchiere:
  • Historische D&A-/Umsatz-Ratio (Segment oder Konzern)
  • Kapitalintensität der Branche
  • Erwartete Investitionszyklen (neue Anlagen → höhere D&A)

Verteilung: Fest oder Dreieck (D&A ist relativ stabil).

Fade-Terminal-Wert (optional):
  → Bei reifem Asset-Bestand sinkt D&A/Umsatz langfristig.
  → Terminal-D&A% = Maintenance-Level.

────────────────────────────────────────────────────────────────────────
2.6  Effektiver Steuersatz (%)
────────────────────────────────────────────────────────────────────────
Recherchiere:
  • Historischer effektiver Konzern-ETR (3 J)
  • Sitzland und lokaler Körperschaftssteuersatz
  • Sondereffekte (Verlustvorträge, Pillar-Two 15 %)
  • Langfristig nachhaltiger Steuersatz

Verteilung: Fest (typisch); Dreieck bei Steuerreform-Unsicherheit.

Fade-Terminal-Wert (optional):
  → Nur wenn regulatorische Änderungen erwartet werden.

────────────────────────────────────────────────────────────────────────
2.7  CAPEX als % vom Umsatz
────────────────────────────────────────────────────────────────────────
Recherchiere:
  • Historische CAPEX/Umsatz-Ratio (Trend?)
  • Management-Guidance zu Investitionsplänen
  • Branchenvergleich (Software ~3–5 %, Telko ~15–20 %, Industrie ~5–8 %)
  • Maintenance vs. Growth CAPEX getrennt (falls verfügbar)

Verteilung: PERT oder Dreieck.

Fade-Terminal-Wert (dringend empfohlen):
  → Terminal-CAPEX% = Erhaltungsinvestitionen (oft deutlich unter
    aktuellem Wert, wenn Growth-Phase endet).
  → Faustregel: Maintenance CAPEX ≈ D&A (langfristig).

────────────────────────────────────────────────────────────────────────
2.8  NWC-Veränderung als % der Umsatzveränderung
────────────────────────────────────────────────────────────────────────
Recherchiere:
  • Historische NWC/Umsatz-Quote und Veränderung
  • Branchencharakteristik:
      Software/Digital 0–5 % | Handel/Industrie 10–20 % | Bau 15–25 %
  • Cash-Conversion-Cycle des Segments

Verteilung: Fest oder PERT.

Fade-Terminal-Wert (optional):
  → Ändert sich die NWC-Intensität bei Shift zu anderem Produktmix?

────────────────────────────────────────────────────────────────────────
2.9  WACC (%)
────────────────────────────────────────────────────────────────────────
Berechne segmentspezifisch und ZEIGE die vollständige Herleitung:

a) k_e (CAPM):
   • Risikofreier Zins: Aktuelle 10Y-Staatsanleihe des Referenzmarktes
   • ERP: Damodaran Equity Risk Premium für relevanten Markt
   • Beta: 3–5 Pure-Play-Peers → Levered Beta → Unlever (Hamada) →
     Re-lever mit Kapitalstruktur von [UNTERNEHMEN]
   • Ggf. Size Premium / Country Risk Premium

b) k_d:
   • Credit Rating → Credit Spread
   • Oder: Durchschnittl. Zinsaufwand / Finanzschulden

c) Kapitalstruktur:
   • E/V und D/V auf Marktwertbasis
   • Zielkapitalstruktur falls kommuniziert

d) Tax Shield auf Fremdkapital

WACC = (E/V · k_e) + (D/V · k_d · (1 − t))

Verteilung: Normal oder PERT (σ typisch 0,5–2,0 pp).

────────────────────────────────────────────────────────────────────────
2.10  Terminal-Value-Methode
────────────────────────────────────────────────────────────────────────

Empfehle PRO SEGMENT eine der beiden Methoden und begründe:

OPTION A – Gordon Growth Model:
  → TV-Wachstumsrate g (**stochastisch – als Verteilung angeben!**):
      Benchmark = langfristiges nominales BIP-Wachstum (2–3 %)
      Muss DEUTLICH < WACC sein (Faustregel: g < WACC − 3pp)
      Für schrumpfende Segmente: 0 % oder negativ
  Verteilung: PERT (z. B. Min=0 %, Mode=2 %, Max=3 %).
  → Die TV-Wachstumsrate wird im Modell DIREKT als Verteilung
    eingegeben (gleiche 6 Typen wie alle anderen Parameter).
    So wird die TV-Unsicherheit in der MC-Simulation erfasst.

OPTION B – Exit Multiple:
  → EV/EBITDA-Multiple (**stochastisch – als Verteilung angeben!**):
      Aktuelle Trading Multiples der Peer Group
      Historische M&A-Transaktionsmultiples
      Langfristiger Branchendurchschnitt
  Verteilung: PERT oder Dreieck (Bear/Base/Bull).
  → Das Exit-Multiple wird im Modell DIREKT als Verteilung
    eingegeben (gleiche 6 Typen wie alle anderen Parameter).
    So wird die Multiple-Unsicherheit in der MC-Simulation erfasst.

Leitlinie:
  • Stabile Cashflows → Gordon Growth
  • Zyklisch / M&A-aktiv → Exit Multiple
  • Tech (Margentestung) → Gordon Growth

────────────────────────────────────────────────────────────────────────
2.11  Fade-Geschwindigkeit λ
────────────────────────────────────────────────────────────────────────
Empfehle einen λ-Wert für das Fade-Modell des Segments:
  • 0,2–0,3 = langsam (langfristige strukturelle Shifts)
  • 0,5     = mittel  (Standard-Empfehlung)
  • 0,8–1,5 = schnell (kurzfristige Normalisierung)
FEST. Gleicher λ gilt für alle Parameter des Segments.

Begründe, ob der Übergang schnell oder langsam sein sollte.


═══════════════════════════════════════════════════════════════════════════
SCHRITT 3 · CORPORATE BRIDGE (ERWEITERTE EQUITY-BRÜCKE)
═══════════════════════════════════════════════════════════════════════════

Die Equity Bridge berechnet:
  Equity = Σ EV_i − PV(Holdingkosten) − Net Debt − Minderheiten
           − Pensionen + Nicht-operative Assets + Beteiligungen

Alle 8 Posten können im Modell als FEST oder als VERTEILUNG eingegeben
werden. Recherchiere und schätze:

────────────────────────────────────────────────────────────────────────
3.1  Jährliche Holdingkosten (Mio.)
────────────────────────────────────────────────────────────────────────
• Nicht segmentzugeordnete Kosten aus der Segmentüberleitung
  (Corporate/Überleitung/Sonstiges)
• Inkl. Vorstandsvergütung, zentrale IT, Konzernfunktionen
• Quelle: Segmentüberleitung im Geschäftsbericht

────────────────────────────────────────────────────────────────────────
3.2  Diskontierungssatz Holdingkosten (%)
────────────────────────────────────────────────────────────────────────
• Typisch: Konzern-WACC (Perpetuity der Holdingkosten)
• FEST oder als Verteilung (z. B. PERT um den WACC-Punktwert),
  wenn die Unsicherheit des Diskontierungssatzes modelliert werden soll.

────────────────────────────────────────────────────────────────────────
3.3  Nettoverschuldung / Net Debt (Mio.)
────────────────────────────────────────────────────────────────────────
• Net Debt = Finanzschulden (kurz- + langfristig)
           − Cash & Äquivalente − kurzfristige Finanzanlagen
• KEINE Pensionen oder Leasingverbindlichkeiten hier einrechnen
  (diese sind separate Posten in 3.6)
• Quelle: Letzte berichtete Bilanz
• Ggf. als Verteilung bei erwartetem Schuldenabbau oder M&A

────────────────────────────────────────────────────────────────────────
3.4  Aktien ausstehend (Mio.) – voll verwässert
────────────────────────────────────────────────────────────────────────
• Basic Shares + Verwässerung durch Optionen (Treasury Stock Method),
  Wandelanleihen, RSUs
• Quelle: Geschäftsbericht → Ergebnis je Aktie (verwässert)
• Ggf. als Verteilung bei laufenden Rückkaufprogrammen

────────────────────────────────────────────────────────────────────────
3.5  Minderheitsanteile (Mio.) → ABZUG
────────────────────────────────────────────────────────────────────────
• Buchwert der Anteile Dritter an konsolidierten Tochtergesellschaften
• Quelle: Bilanz → Eigenkapital → „Non-controlling Interests"
• Falls 0 oder nicht vorhanden → Fest: 0

────────────────────────────────────────────────────────────────────────
3.6  Pensionsrückstellungen (Mio.) → ABZUG
────────────────────────────────────────────────────────────────────────
• Netto-Pensionsverpflichtung = DBO − Plan Assets
• Quelle: Anhang „Leistungen an Arbeitnehmer"
• Besonders relevant bei europäischen Industrieunternehmen
• Falls 0 oder nicht vorhanden → Fest: 0

────────────────────────────────────────────────────────────────────────
3.7  Nicht-operative Assets (Mio.) → ZUSCHLAG
────────────────────────────────────────────────────────────────────────
• Equity-Method-Beteiligungen, überschüssige Immobilien,
  langfristige Finanzanlagen, die nicht zum operativen Kern gehören
• Quelle: Bilanz → Langfristige Vermögenswerte (nicht-operativ)
• Falls 0 → Fest: 0

────────────────────────────────────────────────────────────────────────
3.8  Assoziierte Unternehmen / Beteiligungen (Mio.) → ZUSCHLAG
────────────────────────────────────────────────────────────────────────
• Equity-Buchwert strategischer Minderheitsbeteiligungen
• Quelle: Anhang „Anteile an assoziierten Unternehmen"
• Falls 0 → Fest: 0

⚠️  Für jeden Posten: Entscheide ob FEST oder eine Verteilung
angebracht ist. Verteilung empfohlen bei: erwartetem Schuldenabbau,
laufenden Rückkäufen, unsicherer Pensionsbewertung, oder wenn der
Marktwert von Beteiligungen stark vom Buchwert abweicht.


═══════════════════════════════════════════════════════════════════════════
SCHRITT 4 · CROSS-SEGMENT-KORRELATION
═══════════════════════════════════════════════════════════════════════════

Das Modell verwendet eine Gauss-Copula zur Modellierung stochastischer
Abhängigkeit zwischen Segmenten. Schätze für JEDES Segmentpaar (i, j)
mit i < j:

  • Korrelationskoeffizient  ρ_ij ∈ [−1, 1]
  • Begründung (1–2 Sätze)

Orientierungshilfe:

| Beziehung                                          | Typisches ρ   |
|----------------------------------------------------|---------------|
| Gleiche Branche, gleiche Region                    | 0,6 – 0,9    |
| Gleiche Branche, andere Region                     | 0,3 – 0,6    |
| Komplementär (z. B. Hardware + Services)           | 0,2 – 0,5    |
| Diversifiziert (z. B. Industrie + Finanz)          | 0,0 – 0,3    |
| Gegenläufig (z. B. Upstream + Downstream Öl)       | −0,2 – 0,1   |

Beachte: Die Korrelationsmatrix muss positiv semi-definit sein.
Das Modell validiert dies automatisch.


═══════════════════════════════════════════════════════════════════════════
SCHRITT 5 · SIMULATIONSPARAMETER
═══════════════════════════════════════════════════════════════════════════

Empfehle:
  • Anzahl MC-Iterationen: (Standard: 50.000 für robuste Ergebnisse)
  • Random Seed: 42 (für Reproduzierbarkeit)
  • Mid-Year Convention: Ja/Nein (Standard: Ja – da Cashflows
    unterjährig anfallen)


═══════════════════════════════════════════════════════════════════════════
SCHRITT 6 · AUSGABEFORMAT
═══════════════════════════════════════════════════════════════════════════

Liefere die Ergebnisse in EXAKT den folgenden Tabellen, damit sie
direkt in die Streamlit-App eingegeben werden können.

Für jedes Segment eine eigene Tabelle:

### SEGMENT [N]: [Name]
| Parameter | Verteilung | Min | Mode | Max | μ | σ | Terminal | λ | Begründung |
|---|---|---|---|---|---|---|---|---|---|
| Basisumsatz (Mio.) | FEST: [Wert] | – | – | – | – | – | – | – | [Quelle + GJ] |
| Prognosejahre | FEST: [N] | – | – | – | – | – | – | – | [Warum N Jahre?] |
| Umsatzwachstum (%) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | [g_term] | [λ] | [Herleitung: Hist. CAGR, Konsensus, TAM] |
| EBITDA-Marge (%) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | [m_term] | – | [Hist. Trend, Peer-Median, Skaleneffekte] |
| D&A (% Umsatz) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | [d_term] | – | [Hist. Ratio, Kapitalintensität] |
| Steuersatz (%) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | [t_term] | – | [ETR-Analyse, Jurisdiktion] |
| CAPEX (% Umsatz) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | [c_term] | – | [Maintenance vs. Growth, Guidance] |
| NWC (% ΔUmsatz) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | [n_term] | – | [CCC, Branchenvergleich] |
| WACC (%) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | – | – | [CAPM-Herleitung komplett zeigen!] |
| TV-Methode | [Gordon Growth / Exit Multiple] | – | – | – | – | – | – | – | [Warum diese Methode?] |
| TV-Wachstum (%) | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | – | – | [BIP-Benchmark, WACC-Spread] ★ |
| oder: Exit-Multiple | [Typ]: … | [lo] | [mode] | [hi] | [μ] | [σ] | – | – | [Peer-Multiples, M&A-Comps] ★ |

★ **Wichtig:** TV-Wachstum und Exit-Multiple werden im Modell als
  vollständige Verteilung eingegeben (alle 6 Typen möglich).
  PERT ist hier besonders empfehlenswert, um die Unsicherheit der
  Terminal-Value-Annahmen abzubilden. NICHT als Fest eingeben,
  außer bei extremer Sicherheit.

Hinweise zur Tabelle:
  • „Terminal"-Spalte: Langfristiger Zielwert für den Fade. „–" wenn kein Fade.
  • „λ"-Spalte: Nur in Zeile Umsatzwachstum (gilt für alle Parameter
    des Segments). Alle anderen Zeilen „–".
  • Verteilungstypen: FEST, Normal, Lognormal, Dreieck, Gleichverteilung, PERT.
  • Nicht benötigte Spalten mit „–" füllen (z. B. kein μ/σ bei PERT).

### CORPORATE BRIDGE
| Posten | Wert / Verteilung | Wirkung | Begründung |
|---|---|---|---|
| Holdingkosten p.a. (Mio.) | [Wert oder Verteilung] | Abzug (PV) | [Quelle: Segmentüberleitung, GJ] |
| Diskontierung Holding (%) | FEST: [Wert] | – | [= Konzern-WACC weil …] |
| Nettoverschuldung (Mio.) | [Wert oder Verteilung] | Abzug | [Berechnung: Schulden − Cash zeigen] |
| Aktien (Mio., verwässert) | [Wert oder Verteilung] | Divisor | [Quelle, Verwässerungsmethode] |
| Minderheitsanteile (Mio.) | [Wert oder 0] | Abzug | [Quelle oder „nicht vorhanden"] |
| Pensionsrückstellungen (Mio.) | [Wert oder 0] | Abzug | [DBO − Plan Assets zeigen] |
| Nicht-operative Assets (Mio.) | [Wert oder 0] | Zuschlag | [Quelle: welche Assets konkret?] |
| Beteiligungen (Mio.) | [Wert oder 0] | Zuschlag | [Quelle: welche Beteiligungen?] |

### KORRELATIONSMATRIX
| Segment i | Segment j | ρ_ij | Begründung |
|---|---|---|---|
| [Seg 1] | [Seg 2] | [ρ] | [Warum diese Höhe?] |
| … | … | … | … |

### SIMULATIONSPARAMETER
| Parameter | Wert |
|---|---|
| MC-Iterationen | [z. B. 50.000] |
| Random Seed | [z. B. 42] |
| Mid-Year Convention | [Ja / Nein] |


═══════════════════════════════════════════════════════════════════════════
SCHRITT 7 · PLAUSIBILITÄTS- & QUALITÄTSCHECKS
═══════════════════════════════════════════════════════════════════════════

Führe ALLE folgenden 9 Checks durch und dokumentiere das Ergebnis
explizit (✅ bestanden / ⚠️ Anpassung nötig / ❌ Problem).
Bei ⚠️ oder ❌: Parameter oben anpassen und die NEUE Begründung ergänzen.

1. **Umsatz-Cross-Check**
   Σ Basisumsätze aller Segmente ≈ berichteter Konzernumsatz?
   Abweichung < 5 % → ✅, sonst erklären.

2. **Margin-Cross-Check**
   Gewichteter Durchschnitt der EBITDA-Margen ≈ berichtete
   Konzern-EBITDA-Marge?

3. **WACC-Plausibilität**
   Liegt jeder segmentspezifische WACC in einem plausiblen Bereich
   für Branche, Bonität und Risikofreizins-Umfeld?

4. **TV-Growth ≪ WACC**
   Für JEDES Gordon-Growth-Segment prüfen: g < WACC − 3 pp.
   Falls nicht → TV-Wachstum senken und begründen.

5. **TV/EV-Ratio (Vorab-Schätzung)**
   Schätze grob PV(TV) / EV je Segment.
     < 60 % → ✅ nachhaltig
     60–75 % → ⚠️ akzeptabel, aber Terminal-Annahmen hinterfragen
     > 75 % → ❌ aggressive Annahmen → Parameter anpassen

6. **Implied ROIC**
   Berechne je Segment:
     ROIC ≈ NOPAT-Marge × (1 + g) / Reinvestment Rate
   Plausibel für die Branche?
     Software 25–50 % | Industrie 10–20 % | Handel 8–15 %
   Falls unplausibel → Parameter anpassen.

7. **Fade-Konsistenz**
   Konvergieren alle Terminal-Werte (Marge, CAPEX, NWC) zu
   branchenüblichen Langfristwerten? Jede Fade-Richtung
   ökonomisch sinnvoll? (z. B. CAPEX sinkt von Growth → Maintenance)

8. **Korrelationslogik**
   Paarweise Korrelationen konsistent mit Geschäftslogik?
   Gleicher Endmarkt → höher. Matrix positiv semi-definit?

9. **Impliziter Fair Value (Überschlagsrechnung)**
   Equity ≈ (Σ EV − PV(Holding) − Net Debt − Minority − Pension
             + Non-Op + Associates) / Aktien
   Vergleiche mit aktuellem Kurs → Größenordnung plausibel?
   Premium/Discount in % angeben.

Liefere die Ergebnisse in dieser Tabelle:

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

Liste ALLE verwendeten Quellen auf, gruppiert nach Typ:

| Typ | Quelle | Datum / GJ |
|---|---|---|
| Geschäftsbericht | [Unternehmen] Annual Report [GJ] | [Datum] |
| Analystenberichte | [Bloomberg Consensus, LSEG I/B/E/S etc.] | [Abruf] |
| Branchenreports | [Gartner, IDC, McKinsey etc.] | [Datum] |
| Damodaran | [Betas, ERP, Multiples – pages.stern.nyu.edu/~adamodar/] | [Update] |
| Marktdaten | [Anleiherenditen, CDS, Credit Ratings] | [Stichtag] |
| Sonstige | … | … |


═══════════════════════════════════════════════════════════════════════════
ZUSAMMENFASSUNG (EXECUTIVE SUMMARY)
═══════════════════════════════════════════════════════════════════════════

Schließe mit einer Executive Summary (max. 5 Sätze):
  • Anzahl Segmente und Berichtswährung
  • Wichtigste Werttreiber-Unsicherheiten (wo sind Verteilungen
    besonders breit?)
  • Erwartete Bewertungs-Richtung: Aufschlag oder Abschlag?
  • Empfehlung zur Interpretation der Monte-Carlo-Ergebnisse
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

Das Ergebnis kann direkt in die Streamlit-App eingegeben werden – jede Tabellenzeile
entspricht einem Eingabefeld in der App.
