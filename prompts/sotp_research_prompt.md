# SOTP Monte-Carlo DCF – Research & Estimation Prompt

> **Anleitung:** Ersetze jeden Platzhalter `[UNTERNEHMEN]` durch den Firmennamen (z. B. „Siemens AG"). Übergib den Prompt an ein LLM **mit Web-/Datenbank-Zugriff** oder nutze ihn als strukturierte Rechercheleitlinie.

---

## DER PROMPT

````
Du bist ein Senior Equity Research Analyst (CFA, 15+ Jahre Erfahrung in
fundamentaler Unternehmensbewertung). Deine Aufgabe:

  Recherchiere und schätze SÄMTLICHE Parameter für ein
  Sum-of-the-Parts (SOTP) Monte-Carlo-DCF-Modell
  für [UNTERNEHMEN].

Die Ergebnisse werden als JSON-Block direkt in eine Streamlit-App importiert. Halte dich
EXAKT an die unten definierten Formate und Einheiten.

STRIKTE AUSGABEREGELN (VERPFLICHTEND):
1. Gib ausschließlich die geforderten Tabellen und Blöcke aus. Alle App-Werte
   stehen NUR im JSON-Block aus Schritt 6 (am Ende der Antwort).
2. Keine Einleitung, keine Disclaimer, kein zusätzlicher Fließtext.
3. Einheiten strikt einhalten: % als Prozentwert, absolute Werte in Mio.
4. Verwende exakt die vorgegebenen Spaltennamen, JSON-Keys und Reihenfolge.
5. Fülle nicht genutzte Tabellenfelder mit „–“; im JSON lässt du sie weg.
6. Verwende nur erlaubte Verteilungstypen und korrekte Parameter.
7. Nenne in jeder Begründung mindestens eine belastbare Quelle.
8. Markiere Konzernannahmen explizit mit „≈ Konzernannahme“.
9. Erfinde keine Daten; kennzeichne fehlende Daten transparent.
10. Verwende in Begründungen Kurzverweise [Q1], [Q2], … gemäß Schritt 8.
11. Stichtag statt Geschäftsjahr: Basisumsätze sind LTM bis zum letzten
    berichteten Quartal, Bridge-Posten vom selben Bilanzstichtag.
12. Aktienvergütung (SBC) ist Aufwand: Margen enthalten SBC. Die Non-GAAP-
    Sicht ohne SBC lieferst du als Szenario (Schritt 7b), nicht als Basis.
13. Kennzeichne jede Quelle mit ihrem Prüfstatus: [primär geprüft],
    [primär, teilweise] oder [sekundär] (Schritt 8).


╔═══════════════════════════════════════════════════════════════════════╗
║  MODELLARCHITEKTUR – LIES DIES VOLLSTÄNDIG, BEVOR DU BEGINNST        ║
╚═══════════════════════════════════════════════════════════════════════╝

Das Modell bewertet jedes Segment einzeln über einen FCFF-basierten DCF
und aggregiert die Segmentwerte dann zu einem Gesamtunternehmenswert.
Verstehe die exakte Berechnungskette – JEDER Parameter, den du lieferst,
fließt in diese Formeln ein.


────────────────────────────────────────────────────────────────────────
A · FCFF-BERECHNUNGSKETTE (exakte Formeln im Modell)
────────────────────────────────────────────────────────────────────────

Für jede Simulation und jedes Jahr t ∈ {1, …, T}:

  1. Revenue_t (Konstant):   Base_Revenue × (1 + g)^t
     Revenue_t (Fade):       Revenue_{t−1} × (1 + g_t)
       mit  g_t = g_terminal + (g_initial − g_terminal) · e^(−λ · t)

  2. EBITDA_t              = Revenue_t × EBITDA-Marge
  3. D&A_t                 = Revenue_t × D&A%
  4. EBIT_t                = EBITDA_t − D&A_t
  5. Steuern_t             = max(EBIT_t, 0) × Steuersatz
     (Steuern nur auf positives EBIT!)
  6. NOPAT_t               = EBIT_t − Steuern_t
  7. CAPEX_t               = Revenue_t × CAPEX%
  8. ΔNWC_t                = (Revenue_t − Revenue_{t−1}) × NWC%
     ⚠️ NWC% ist der Anteil der UMSATZVERÄNDERUNG, NICHT des Umsatzes!
     Revenue_0 = Base_Revenue

  9. FCFF_t = NOPAT_t + D&A_t − CAPEX_t − ΔNWC_t

Terminal Value (nach Jahr T):
  Gordon Growth:  TV = FCFF_T × (1 + g_TV) / (WACC − g_TV)
  Gordon Growth mit Parameter-Fade:
                  TV = FCFF_{T+1} / (WACC − g_TV)
                  FCFF_{T+1} mit Revenue_T × (1 + g_TV) und den TERMINAL-
                  Werten (Marge, D&A, Steuer, CAPEX, NWC), nicht mit den
                  Werten, die der Fade in Jahr T erreicht hat.
                  → Terminal-Werte sind der Steady State, der den
                    Endwert trägt. Wähle sie entsprechend sorgfältig.
  Exit Multiple:  TV = EBITDA_T × Multiple

Diskontierung:
  Mid-Year Convention (Standard):
    PV(FCFF_t) = FCFF_t / (1 + WACC)^(t − 0,5)
  PV(TV)       = TV / (1 + WACC)^T

  Segment-EV = Σ PV(FCFF_t) + PV(TV)


────────────────────────────────────────────────────────────────────────
B · EQUITY BRIDGE (vom Enterprise Value zum Aktienkurs)
────────────────────────────────────────────────────────────────────────

  Equity = Σ Segment-EV_i
         − PV(Holdingkosten)       [= Holdingkosten p.a. / Diskontierungssatz]
         − Nettoverschuldung
         − Minderheitsanteile
         − Pensionsrückstellungen
         + Nicht-operative Assets
         + Beteiligungen

  Kurs je Aktie = Equity / Aktien ausstehend (voll verwässert)


────────────────────────────────────────────────────────────────────────
C · VERTEILUNGSTYPEN & EINGABEFORMAT
────────────────────────────────────────────────────────────────────────

Die App unterstützt 6 Verteilungstypen. Für JEDEN stochastischen
Parameter musst du den Typ und die zugehörigen Werte angeben.

  ┌────────────────────┬────────────────────────────────────────────────┐
  │ Typ                │ Was du in der App eingibst                     │
  ├────────────────────┼────────────────────────────────────────────────┤
  │ Fest               │ Wert                                          │
  │ Normalverteilung   │ μ (Mittelwert), σ (Std.-Abw.)                │
  │ Lognormalverteilung│ Gewünschter Mittelwert, Gewünschte Std.-Abw. │
  │                    │ ⚠️ NICHT μ/σ der unterliegenden Normalvert.!  │
  │                    │ Die App konvertiert intern automatisch.        │
  │ Dreiecksverteilung │ Min, Mode, Max                                │
  │ Gleichverteilung   │ Min, Max                                      │
  │ PERT-Verteilung    │ Min, Mode, Max  (λ = 4, intern fest)          │
  └────────────────────┴────────────────────────────────────────────────┘

  ⚠️ LOGNORMAL: Du gibst den ERWARTETEN Mittelwert und die ERWARTETE
     Standardabweichung der Zielverteilung direkt ein. Beispiel:
     EBITDA-Marge soll ~20 % ± 3 % sein → Eingabe: μ = 20, σ = 3.
     Die App berechnet daraus die darunterliegenden Normal-Parameter.

  Entscheidungshilfe Verteilungswahl:
    → Hohe Sicherheit, historischer Fakt      → Fest
    → Symmetrische Unsicherheit               → Normal
    → Nur positive Werte, rechtsschief        → Lognormal
    → Min/Mode/Max bekannt, moderate Gewichte → PERT ★ BEVORZUGT
    → Min/Mode/Max bekannt, Extremwerte fair  → Dreieck
    → Nur Grenzen bekannt, kein Mode          → Gleichverteilung

  ⚠️ PERT vor Dreieck bevorzugen (weniger Gewicht auf Extremwerte).
  ⚠️ Bei Unsicherheit: BREITERE Verteilung wählen – das ist der
     Mehrwert einer MC-Simulation gegenüber einem Punkt-DCF.


────────────────────────────────────────────────────────────────────────
D · EINHEITEN – KRITISCH!
────────────────────────────────────────────────────────────────────────

  • Alle Prozentangaben als %, NICHT als Dezimalzahl.
    Beispiel: 5 % Wachstum → Eingabe: 5.0  (NICHT 0.05)
    Die App konvertiert intern durch Division mit 100.
  • Absolute Werte in Mio. Berichtswährung.
  • Berichtswährung von [UNTERNEHMEN] konsistent nutzen und EINMALIG
    am Anfang nennen.


────────────────────────────────────────────────────────────────────────
E · FADE-MODELL (exponentiell, universell)
────────────────────────────────────────────────────────────────────────

  Jeder Werttreiber kann von einem initialen zu einem terminalen Wert
  konvergieren:

    p(t) = p_terminal + (p_initial − p_terminal) · e^(−λ · t)

  • p_initial = gesampelter Startwert (aus der Verteilung)
  • p_terminal = langfristiges Gleichgewicht (eigene Verteilung)
  • λ = Fade-Geschwindigkeit (EIN λ pro Segment, gilt für ALLE
    Parameter dieses Segments gleichzeitig)

  Fade unterstützt für:
    Umsatzwachstum, EBITDA-Marge, D&A%, Steuersatz, CAPEX%, NWC%

  ⚠️ WACC hat KEIN Fade. WACC wird als konstant über den gesamten
     Prognosezeitraum gesampelt.


────────────────────────────────────────────────────────────────────────
F · STOCHASTIK & SAMPLING
────────────────────────────────────────────────────────────────────────

  Monte-Carlo-Simulation (Standard: 50 000 Iterationen).

  2 Sampling-Strategien:
    ┌─────────────────────────┬──────────────────────────────────────┐
    │ Strategie               │ Eigenschaften                        │
    ├─────────────────────────┼──────────────────────────────────────┤
    │ Pseudo-Random (Standard)│ Klassisches MC, NumPy-RNG            │
    │ Quasi-MC (Sobol)        │ Scrambled Sobol → schnellere Konverg.│
    └─────────────────────────┴──────────────────────────────────────┘

  Empfehlung: Sobol bei ≤ 7 stochastischen Parametern pro Segment
  (optimale Diskrepanz). Pseudo-Random, wenn maximale Flexibilität bei
  vielen korrelierten Segmenten nötig.


────────────────────────────────────────────────────────────────────────
G · KORRELATIONSSTRUKTUR
────────────────────────────────────────────────────────────────────────

  CROSS-SEGMENT-KORRELATION (Gauss-Copula):
    Ein Korrelationskoeffizient ρ pro Segmentpaar korreliert ALLE
    stochastischen Parameter beider Segmente gleichzeitig, ohne die
    Marginalverteilungen zu verzerren.
    Eingabe: N×N-Matrix in der App. Diagonale = 1 (fest, nicht editierbar).
    Oberes Dreieck editierbar, unteres wird automatisch gespiegelt.
    Validierung: positive Semidefinitheit wird automatisch geprüft.

  INTRA-SEGMENT-COPULA (7×7-Gauss-Copula, OPTIONAL):
    Modelliert Abhängigkeiten INNERHALB eines Segments zwischen:
      [Umsatzwachstum, EBITDA-Marge, D&A, Steuersatz, CAPEX, NWC, WACC]

    Eingabe: 7×7-Matrix in der App. Diagonale = 1 (fest).
    Oberes Dreieck editierbar, unteres wird automatisch gespiegelt.

    Standard-Defaults (überschreibbar):
                        Wachst.  EBITDA   D&A    Steuer  CAPEX   NWC    WACC
      Wachstum           1.00    0.30    0.10    0.00    0.15    0.35   −0.10
      EBITDA-Marge        0.30    1.00    0.20    0.00    0.25    0.10   −0.20
      D&A                 0.10    0.20    1.00    0.00    0.70    0.05    0.00
      Steuersatz          0.00    0.00    0.00    1.00    0.00    0.00    0.10
      CAPEX               0.15    0.25    0.70    0.00    1.00    0.10    0.05
      NWC                 0.35    0.10    0.05    0.00    0.10    1.00    0.00
      WACC               −0.10   −0.20    0.00    0.10    0.05    0.00    1.00

    ⚠️ NUR aktivieren, wenn du fundierte Schätzungen für die paarweisen
       Abhängigkeiten liefern kannst. Die App funktioniert auch ohne.


────────────────────────────────────────────────────────────────────────
H · QUALITÄTSMETRIKEN (automatisch berechnet – für deine Checks)
────────────────────────────────────────────────────────────────────────

  Die App berechnet automatisch pro Segment:

  • TV/EV-Ratio = PV(Terminal Value) / Segment-EV
    Ziel: < 60 %. > 75 % → Prognosezeitraum zu kurz oder FCFF-Margin zu niedrig.

  • Implied ROIC (exakte Formel im Modell):
      NOPAT-Marge = (EBITDA% − D&A%) × (1 − Steuersatz)
      Reinvest-Marge = CAPEX% − D&A% + NWC% × g / (1 + g)
      ROIC = g × NOPAT-Marge / Reinvest-Marge

    ⚠️ Das ist NICHT die klassische Lehrbuchformel! Das Modell
       leitet ROIC aus der Steady-State-Identität g = ROIC × b ab:
       ROIC = g / b,  wobei b = Reinvest-Marge / NOPAT-Marge.
    Plausible Bereiche: Software 25–50 % | Industrie 10–20 % | Handel 8–15 %

  • Reinvestment Rate = Reinvest-Marge / NOPAT-Marge
    (Anteil des NOPAT, der reinvestiert wird)

  • Composite Quality Score (0–100):
      TV/EV-Sub (0–25) + Konvergenz-Sub (0–25)
      + Sensitivity-Sub (0–25) + Dispersion-Sub (0–25)


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
  e) Sektorklassifikation:
      Technologie | Konsumgüter | Gesundheit | Finanzen | Energie |
      Industrie | Grundstoffe | Immobilien | Telekommunikation |
      Versorger | Sonstige
3. Sortiere absteigend nach Umsatz.
4. SONDERFALL nur ein berichtetes Segment: Teile nach den berichteten
   Erlösarten der GuV (z. B. Produkte / Subscriptions / Wartung). Verteile
   die Kosten über die Bruttomargen je Erlösart, kennzeichne die Margen mit
   „≈ Konzernannahme" und prüfe sie in Check 2 gegen die Konzernmarge.
5. SONDERFALL Teile ohne sinnvollen DCF (Verlustbringer mit Marktbewertung,
   z. B. Venture-Einheiten; Finanzbeteiligungen): KEIN Segment, sondern
   Bridge-Posten (Schritt 3.7/3.8). Bewertung zur letzten Finanzierungsrunde
   bzw. zum Marktwert × Anteil, abzüglich latenter Steuer auf Buchgewinne.
   Konsolidiertes Cash dieser Einheit nicht doppelt zählen.

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
App-Feld: „Basisumsatz (Mio., Jahr 0)"
• LTM-Segmentumsatz bis zum letzten berichteten Quartal, damit Jahr 1
  im Modell den nächsten 12 Monaten ab Stichtag entspricht:
    LTM = GJ + YTD(aktuell) − YTD(Vorjahr)  (oder Summe der 4 Quartale)
  Herleitung in der Begründung zeigen. Liegt das GJ-Ende < 3 Monate
  zurück, ist das GJ selbst die LTM-Basis.
• Zeitraum in "metadata.basis_periode" angeben (z. B. "LTM Q4/25–Q3/26").
• IMMER Fest (historischer Fakt).
• Quelle: Segmentberichterstattung / Quartalsberichte.

────────────────────────────────────────────────────────────────────────
2.2  Prognosezeitraum (Jahre)
────────────────────────────────────────────────────────────────────────
App-Feld: „Detailzeitraum (Jahre)"  [1–30, ganzzahlig]
• Reife/stabile Segmente → 5 Jahre
• Wachstumssegmente → 7–10 Jahre (bis Steady-State plausibel)
• Zyklische Segmente → voller Konjunkturzyklus (5–7 Jahre)
• IMMER Fest.

────────────────────────────────────────────────────────────────────────
2.3  Umsatzwachstum (%)
────────────────────────────────────────────────────────────────────────
App-Feld: „Umsatzwachstum" (Verteilungsinput, Einheit: %)
Wachstumsmodell-Auswahl:
  „Konstant (g über alle Jahre gleich)"
  oder
  „Fade-Modell (g konvergiert zum Terminal-Wachstum)"

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

Numerisches Beispiel (bei PERT):
  „Umsatzwachstum: PERT(Min=3, Mode=7, Max=12)"
  bedeutet: Min 3 %, wahrscheinlichstes 7 %, Max 12 %

────────────────────────────────────────────────────────────────────────
2.4  EBITDA-Marge (%)
────────────────────────────────────────────────────────────────────────
App-Feld: „EBITDA-Marge" (Verteilungsinput, Einheit: %)

Recherchiere:
  • Historische Segmentmarge (3–5 J, Trend?)
  • Peer-Group Median (Pure-Play-Vergleiche)
  • Skaleneffekte & operative Leverage
  • Management-Zielvorgaben („Mid-term targets")

Verteilung: PERT bevorzugt. Normal bei Symmetrie.

Rolle im Modell:
  EBITDA = Revenue × EBITDA-Marge
  → Direkt proportional zum FCFF. Eine um 1 pp höhere Marge
    erhöht EBITDA und damit NOPAT und FCFF direkt.

Fade-Terminal (empfohlen):
  → Langfristig nachhaltige EBITDA-Marge.
  → Marge über Peer-Median? → Fade nach unten (Mean Reversion).
  → Wachstumssegment mit Skalierung? → Fade nach oben.

────────────────────────────────────────────────────────────────────────
2.5  D&A als % vom Umsatz
────────────────────────────────────────────────────────────────────────
App-Feld: „D&A (% Umsatz)" (Verteilungsinput, Einheit: %)

Recherchiere:
  • Historische D&A/Umsatz-Ratio (Segment oder Konzern)
  • Kapitalintensität der Branche
  • Investitionszyklen (neue Assets → höhere D&A)

Verteilung: Fest oder Dreieck/PERT (D&A ist relativ stabil).

Rolle im Modell:
  D&A erscheint an ZWEI Stellen:
  1. EBIT = EBITDA − D&A  (senkt den operativen Gewinn → senkt Steuern)
  2. FCFF = NOPAT + D&A − CAPEX − ΔNWC  (wird zum NOPAT addiert,
     da nicht zahlungswirksam)
  → Netto-Effekt: D&A wirkt als Tax Shield. Höhere D&A senkt Steuern.

Fade-Terminal (optional):
  → Bei reifem Asset-Bestand sinkt D&A/Umsatz langfristig zum
    Maintenance-Level.

────────────────────────────────────────────────────────────────────────
2.6  Effektiver Steuersatz (%)
────────────────────────────────────────────────────────────────────────
App-Feld: „Steuersatz" (Verteilungsinput, Einheit: %)

Recherchiere:
  • Historischer effektiver Konzern-ETR (3 J)
  • Körperschaftssteuersatz des Sitzlandes
  • Sondereffekte (Verlustvorträge, Pillar-Two 15 %)
  • Langfristig nachhaltiger ETR

Verteilung: Fest (typisch). Dreieck bei Steuerreform-Unsicherheit.

Rolle im Modell:
  Steuern = max(EBIT, 0) × Steuersatz
  → Steuern fallen NUR auf positives EBIT an. Bei negativem EBIT
    werden keine Steuern berechnet (kein Verlustvortrag modelliert).

Fade-Terminal (optional):
  → Nur bei erwarteten regulatorischen Änderungen.

────────────────────────────────────────────────────────────────────────
2.7  CAPEX als % vom Umsatz
────────────────────────────────────────────────────────────────────────
App-Feld: „CAPEX (% Umsatz)" (Verteilungsinput, Einheit: %)

Recherchiere:
  • Historische CAPEX/Umsatz-Quote (Trend?)
  • Management-Guidance zu Investitionen
  • Branchenvergleich:
      Software ~3–5 % | Telko ~15–20 % | Industrie ~5–8 %
  • Maintenance vs. Growth CAPEX (falls berichtet)

Verteilung: PERT oder Dreieck.

Rolle im Modell:
  CAPEX = Revenue × CAPEX%
  → Wird vom FCFF ABGEZOGEN. Höhere CAPEX = niedrigerer FCFF.

Fade-Terminal (dringend empfohlen):
  → Terminal-CAPEX = Erhaltungsinvestitionen (Maintenance CAPEX).
  → Faustregel: Maintenance CAPEX ≈ D&A langfristig.
  → Growth CAPEX entfällt im Steady State.
  → INVESTITIONSPHASE (CAPEX heute deutlich über historischem Niveau,
    z. B. KI-Rechenzentren): Start-Wert aus Guidance für die nächsten
    12 Monate kalibrieren. Beachte: Jahr-1-Wert = Terminal + (Start −
    Terminal) · e^(−λ), Start also entsprechend höher ansetzen.
    Terminal gegen die CAPEX-Quote VOR der Investitionsphase begründen.
    Die App nutzt EIN λ für Wachstum, Margen und CAPEX – nenne in der
    Begründung, wie lange CAPEX dadurch über Terminal bleibt.
  → D&A-Terminal konsistent wählen: ≈ 0,8–1,0 × CAPEX-Terminal (Check 10).

────────────────────────────────────────────────────────────────────────
2.8  NWC-Veränderung als % der Umsatzveränderung
────────────────────────────────────────────────────────────────────────
App-Feld: „NWC (% ΔUmsatz)" (Verteilungsinput, Einheit: %)

⚠️ ACHTUNG: Dieser Parameter ist NICHT NWC als % des Umsatzes!
   Er ist der Anteil der UMSATZVERÄNDERUNG, der in Net Working Capital
   gebunden/freigesetzt wird.

   Formel im Modell: ΔNWC_t = (Revenue_t − Revenue_{t−1}) × NWC%
   → Bei wachsendem Umsatz: ΔNWC > 0 → Cash-Abfluss (FCFF sinkt)
   → Bei schrumpfendem Umsatz: ΔNWC < 0 → Cash-Zufluss (FCFF steigt)

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
App-Feld: „WACC" (Verteilungsinput, Einheit: %)

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

⚠️ WACC hat KEIN Fade. Er bleibt konstant über alle Jahre.
Hinweis: Die App erzwingt WACC ≥ 0,5 % als Sicherheitsgrenze.

Rolle im Modell:
  WACC dient als Diskontierungssatz für FCFF und Terminal Value.
  Er beeinflusst auch den Terminal Value über den Nenner (WACC − g).
  → Der sensibelste Parameter. 1 pp mehr WACC kann 15–25 % Wertverlust
    bedeuten.

────────────────────────────────────────────────────────────────────────
2.10  TV-Methode & Parameter
────────────────────────────────────────────────────────────────────────
App-Feld: „Methode" (Selectbox)

Empfehle PRO SEGMENT eine der beiden Methoden und begründe:

OPTION A – Gordon Growth Model
  App-Feld: „TV-Wachstumsrate" (Verteilungsinput, Einheit: %)
  • g = langfristiges nachhaltiges Wachstum
  • Formel: TV = FCFF_T × (1 + g) / (WACC − g); mit Parameter-Fade
    FCFF_{T+1} zu Terminal-Werten (siehe A)
  • Benchmark: nominales BIP-Wachstum (2–3 %)
  • MUSS deutlich < WACC sein (Faustregel: g < WACC − 3 pp)
  • Schrumpfende Segmente: 0 % oder negativ
  • ALS VERTEILUNG angeben! PERT bevorzugt.

OPTION B – Exit Multiple
  App-Feld: „Exit-Multiple (EV/EBITDA)" (Verteilungsinput, KEIN %)
  • Formel: TV = EBITDA_T × Multiple
  • Basierend auf: aktuellen Peer-Trading-Multiples,
    historischen M&A-Transaktionsmultiples,
    langfristigem Branchendurchschnitt
  • ALS VERTEILUNG angeben! PERT oder Dreieck.
  ⚠️ Einheit: reiner Multiplikator (z.B. 10.0), NICHT Prozent!

  ⚠️ SONDERFALL: Exit Multiple + Fade-Modell
    Wenn für ein Segment das Fade-Modell UND Exit Multiple gewählt
    werden, muss zusätzlich ein „Langfristiges Umsatzwachstum
    (Fade-Ziel)" angegeben werden:
    App-Feld: „Langfristiges Umsatzwachstum (Fade-Ziel)" (Einheit: %)
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
⚠️ EIN λ pro Segment – NICHT pro Parameter.

Orientierung:
  • 0,2–0,3 = langsam (langfristige strukturelle Shifts)
  • 0,5     = mittel  (Standard-Empfehlung)
  • 0,8–1,5 = schnell (kurzfristige Normalisierung)
  • > 1,5   = sehr schnell (fast sofortige Normalisierung)

Praktische Interpretation:
  Bei λ = 0,5 und T = 5: Nach 2 Jahren ist ~63 % der Differenz abgebaut.
  Bei λ = 1,0 und T = 5: Nach 1 Jahr ist ~63 % der Differenz abgebaut.

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
Diese Terminal-Werte werden mit demselben λ aus 2.11 gefaded.

Empfehlung: Mindestens EBITDA-Marge und CAPEX mit Terminal-Werten
versehen. Diese haben den größten Einfluss auf den Fair Value.

Numerisches Beispiel:
  EBITDA-Marge initial: PERT(15, 20, 25)
  EBITDA-Marge terminal: PERT(18, 22, 28)
  λ = 0,5
  → Start ~20 %, konvergiert über 5 Jahre graduell Richtung ~22 %


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
• App berechnet PV als ewige Rente:  PV = Holdingkosten / Diskontierungssatz
• Einheit: Mio. (absolut), NICHT Prozent

────────────────────────────────────────────────────────────────────────
3.2  Diskontierungssatz für Holdingkosten (%)           → für PV(Holding)
────────────────────────────────────────────────────────────────────────
App-Feld: „Diskontierungssatz (%)" (Verteilungsinput)
• Typisch: Konzern-WACC
• Auch als Verteilung modellierbar
• Einheit: %

────────────────────────────────────────────────────────────────────────
3.3  Nettoverschuldung (Mio.)                           → Abzug
────────────────────────────────────────────────────────────────────────
App-Feld: „Nettoverschuldung (Mio.)" (Verteilungsinput)
• Net Debt = Finanzschulden − Cash & Äquivalente − Finanzanlagen
• KEINE Pensionen oder Leasing hier (→ separate Posten)
• Quelle: Letzte berichtete Bilanz
• Einheit: Mio. (absolut)

────────────────────────────────────────────────────────────────────────
3.4  Aktien ausstehend (Mio., voll verwässert)          → Divisor
────────────────────────────────────────────────────────────────────────
App-Feld: „Aktien ausstehend (Mio.)" (Verteilungsinput)
• Basic Shares + Verwässerung (Treasury Stock Method)
• Verteilung bei laufenden Rückkaufprogrammen
• Einheit: Mio. Stück

────────────────────────────────────────────────────────────────────────
3.5  Minderheitsanteile (Mio.)                          → Abzug
────────────────────────────────────────────────────────────────────────
App-Feld: „Minderheitsanteile (Mio.)"
• Buchwert Non-controlling Interests
• Falls nicht vorhanden → Fest: 0
• Einheit: Mio.

────────────────────────────────────────────────────────────────────────
3.6  Pensionsrückstellungen (Mio.)                      → Abzug
────────────────────────────────────────────────────────────────────────
App-Feld: „Pensionsrückstellungen (Mio.)"
• Netto-Pensionsverpflichtung = DBO − Plan Assets
• Falls nicht vorhanden → Fest: 0
• Einheit: Mio.

────────────────────────────────────────────────────────────────────────
3.7  Nicht-operative Assets (Mio.)                      → Zuschlag
────────────────────────────────────────────────────────────────────────
App-Feld: „Nicht-operative Assets (Mio.)"
• Überschüssige Immobilien, langfristige Finanzanlagen außerhalb
  des operativen Kerns, überschüssiges Cash
• Falls nicht vorhanden → Fest: 0
• Einheit: Mio.

────────────────────────────────────────────────────────────────────────
3.8  Beteiligungen (Mio.)                               → Zuschlag
────────────────────────────────────────────────────────────────────────
App-Feld: „Beteiligungen (Mio.)"
• Equity-Buchwert strategischer Minderheitsbeteiligungen
  (Equity-Method-Beteiligungen)
• Falls nicht vorhanden → Fest: 0
• Einheit: Mio.


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

Eingabe in der App: N×N-Gitter mit number_input-Feldern.
Diagonale = 1,0 (fest). Oberes Dreieck editierbar.
Unteres Dreieck wird automatisch gespiegelt.

⚠️ Die App validiert automatisch positive Semidefinitheit der Matrix.
   Wähle Korrelationen, die gemeinsam konsistent sind.


═══════════════════════════════════════════════════════════════════════════
SCHRITT 4b · INTRA-SEGMENT-COPULA (OPTIONAL)
═══════════════════════════════════════════════════════════════════════════

Die App unterstützt optional eine 7×7-Gauss-Copula INNERHALB jedes
Segments, die Abhängigkeiten zwischen den 7 Werttreibern modelliert:

  [Umsatzwachstum, EBITDA-Marge, D&A, Steuersatz, CAPEX, NWC, WACC]

Falls aktiviert, schätze für jedes Segment die paarweisen
Korrelationskoeffizienten zwischen diesen Parametern.

Eingabe in der App: 7×7-Gitter mit number_input-Feldern.
Diagonale = 1,0 (fest). Oberes Dreieck editierbar.
Unteres Dreieck wird automatisch gespiegelt.
Es gibt vorbelegte Default-Werte (siehe Abschnitt G oben).

Typische Muster:

| Paar                    | Richtung | Typisches ρ | Begründung |
|-------------------------|----------|-------------|------------|
| Wachstum ↔ Marge        | +        | 0,1 – 0,4  | Skaleneffekte |
| Wachstum ↔ CAPEX        | +        | 0,2 – 0,5  | Investition für Wachstum |
| Wachstum ↔ NWC          | +        | 0,2 – 0,4  | Wachstum bindet Working Capital |
| Marge ↔ CAPEX            | +        | 0,1 – 0,3  | Investitionsintensität ermöglicht Margen |
| D&A ↔ CAPEX              | +        | 0,3 – 0,7  | Investition → Abschreibung (stark!) |
| WACC ↔ Marge             | −        | −0,3 – 0,0 | Risikobehaftete Segmente: niedrigere Marge |
| Steuer ↔ andere          | ≈ 0      | 0,0 – 0,1  | Steuer weitgehend extern bestimmt |

⚠️ NUR aktivieren, wenn du fundierte Schätzungen liefern kannst.
   Die App funktioniert auch ohne Intra-Segment-Copula.

Ausgabeformat: 7×7-Matrix pro Segment (oberes Dreieck ausreichend).


═══════════════════════════════════════════════════════════════════════════
SCHRITT 5 · SIMULATIONSPARAMETER
═══════════════════════════════════════════════════════════════════════════

| Parameter               | Empfehlung       | App-Feld             |
|-------------------------|------------------|----------------------|
| MC-Iterationen          | 50 000           | Iterationen          |
| Random Seed             | 42               | Seed (Reproduzierbarkeit) |
| Mid-Year Convention     | Ja (Standard)    | ⏱️ Mid-Year Convention aktiv |
| Sampling-Methode        | Sobol (oder PR)  | Sampling             |

Sampling-Empfehlung:
  • Wenige bis mittlere Dimensionalität → Quasi-MC (Sobol)
  • Hohe Dimensionalität / komplexe Korrelationen → Pseudo-Random


═══════════════════════════════════════════════════════════════════════════
SCHRITT 6 · APP-IMPORT ALS JSON (ein Block, GANZ AM ENDE der Antwort)
═══════════════════════════════════════════════════════════════════════════

Liefere ALLE Werte aus Schritt 1–5 in EINEM ```json-Block. Die App
übernimmt ihn über „📋 Research-Ergebnis einfügen" – nichts wird abgetippt.

Reihenfolge der Antwort:
  Schritt 7 (Checks) → Schritt 7b (Szenarien) → Schritt 8 (Quellen)
  → Executive Summary → JSON-Block
So sind Korrekturen aus den Plausibilitätschecks im JSON bereits enthalten.

JSON-REGELN:
  • Gültiges JSON: doppelte Anführungszeichen, Dezimalpunkt, keine
    Kommentare, keine Tausendertrennzeichen. Nicht genutzte Felder WEGLASSEN.
  • Werte in Anzeige-Einheit: % als Prozentwert (6.5 = 6,5 %), absolute
    Werte in Mio. Lognormal: GEWÜNSCHTER Mittelwert / Std.-Abw.
  • Segment-Index i beginnt bei 0.
  • Verteilungstyp-Strings EXAKT so:
      "Fest (Deterministisch)", "Normalverteilung", "Lognormalverteilung",
      "Dreiecksverteilung", "Gleichverteilung", "PERT-Verteilung"
  • Pro Parameter-Präfix <p> immer <p>_dtype plus NUR die Wert-Keys
    des gewählten Typs:
      Fest       → <p>_fixed
      Normal     → <p>_n_mu, <p>_n_sig
      Lognormal  → <p>_ln_mu, <p>_ln_sig
      Dreieck    → <p>_tri_lo, <p>_tri_mo, <p>_tri_hi
      Gleich     → <p>_uni_lo, <p>_uni_hi
      PERT       → <p>_pert_lo, <p>_pert_mo, <p>_pert_hi

PARAMETER-PRÄFIXE <p>:
  Segment i:  s{i}_rg   Umsatzwachstum      s{i}_em   EBITDA-Marge
              s{i}_da   D&A (% Umsatz)      s{i}_tx   Steuersatz
              s{i}_cx   CAPEX (% Umsatz)    s{i}_nwc  NWC (% ΔUmsatz)
              s{i}_wacc WACC                s{i}_tvg  TV-Wachstum
              s{i}_evm  Exit-Multiple (nur bei Exit-Multiple-Ansatz)
  Fade-Terminals (nur wenn seg_{i}_param_fade = true):
              s{i}_em_term, s{i}_da_term, s{i}_tx_term, s{i}_cx_term, s{i}_nwc_term
  Bridge:     bridge_cc Holdingkosten p.a.     bridge_cd Diskontierungssatz (%)
              bridge_nd Nettoverschuldung      bridge_sh Aktien verwässert (Mio.)
              bridge_mi Minderheitsanteile     bridge_pn Pensionsrückstellungen
              bridge_no Nicht-operative Assets bridge_as Beteiligungen

SEGMENT-FELDER (direkte Werte, keine Verteilung):
  seg_{i}_name         Segmentname
  seg_{i}_basrev       Basisumsatz (Mio.)
  seg_{i}_fyrs         Prognosejahre (Ganzzahl)
  seg_{i}_growth_mode  "Konstant (g über alle Jahre gleich)" oder
                       "Fade-Modell (g konvergiert zum Terminal-Wachstum)"
  seg_{i}_fade_speed   λ (z. B. 0.5)
  seg_{i}_param_fade   true / false
  seg_{i}_tv_method    "Gordon Growth Model (Ewige Rente)" oder "Exit-Multiple-Ansatz"
  seg_{i}_intra_corr   true / false; wenn true zusätzlich seg_{i}_ic_{r}_{c}
                       für r < c (0 Wachstum, 1 EBITDA, 2 D&A, 3 Steuer,
                       4 CAPEX, 5 NWC, 6 WACC)

KORRELATION: corr_{i}_{j} für ALLE Paare inkl. Diagonale (1.0), symmetrisch.
Bei nur einem Segment: "setup_corr_enable": false und "correlation": {}.

QUELLEN (Pflicht – macht die Recherche in der App nachvollziehbar):
  "quellen": je Parameter-Präfix bzw. Segment-Feld EINE Begründung mit
  Kurzverweis, z. B. "s0_rg": "CAGR 2020–25 5,8 % [Q1]; Konsens 2026–28
  6–7 % [Q2]". Pflicht für alle Präfixe aus Schritt 2 und 3 sowie
  seg_{i}_basrev und seg_{i}_fyrs. Die App zeigt sie direkt am Eingabefeld.
  "quellenverzeichnis": Vollzitate aus Schritt 8, Schlüssel "Q1", "Q2", …,
  jeweils mit Prüfstatus am Ende: [primär geprüft] | [primär, teilweise]
  | [sekundär].

  Zweckentfremdete Bridge-Felder (z. B. Steuerrückstellung im Feld
  Pensionen, Vorzugsaktien als Minderheiten) in "quellen" mit „Hier: …"
  beginnen, damit die Belegung am Eingabefeld sichtbar ist.

METADATEN:
  "stichtag" = Datum der Recherche (JJJJ-MM-TT); "saved_at" = dasselbe
  Datum als "JJJJ-MM-TTT00:00:00". "kurs_bei_recherche" = Aktienkurs am
  Stichtag in Berichtswährung (oder weglassen) – die App zeigt ihn im
  Versionsvergleich neben der Bewertung. "basis_periode" = Zeitraum der
  Basisumsätze (z. B. "LTM Q4/25–Q3/26"). "bilanzstichtag" = Datum der
  Bilanz, aus der die Bridge-Posten stammen.

VORLAGE (ein Segment vollständig; weitere Segmente analog mit i = 1, 2, …):

```json
{
  "schema_version": 2,
  "saved_at": "2026-01-15T00:00:00",
  "metadata": {
    "company": "[UNTERNEHMEN]",
    "report_currency": "EUR",
    "stichtag": "2026-01-15",
    "kurs_bei_recherche": 100.0,
    "basis_periode": "LTM Q4/24–Q3/25",
    "bilanzstichtag": "2025-09-30"
  },
  "ui_state": {
    "setup": {
      "setup_n_sim": 50000,
      "setup_seed": 42,
      "setup_n_seg": 1,
      "setup_mid_year": true,
      "setup_ext_bridge": true,
      "setup_sampling": "Pseudo-Random (Standard)",
      "setup_corr_enable": false
    },
    "bridge": {
      "bridge_cc_dtype": "Fest (Deterministisch)", "bridge_cc_fixed": 50,
      "bridge_cd_dtype": "Fest (Deterministisch)", "bridge_cd_fixed": 8.5,
      "bridge_nd_dtype": "PERT-Verteilung", "bridge_nd_pert_lo": 900, "bridge_nd_pert_mo": 1000, "bridge_nd_pert_hi": 1150,
      "bridge_sh_dtype": "Fest (Deterministisch)", "bridge_sh_fixed": 100,
      "bridge_mi_dtype": "Fest (Deterministisch)", "bridge_mi_fixed": 0,
      "bridge_pn_dtype": "Fest (Deterministisch)", "bridge_pn_fixed": 0,
      "bridge_no_dtype": "Fest (Deterministisch)", "bridge_no_fixed": 0,
      "bridge_as_dtype": "Fest (Deterministisch)", "bridge_as_fixed": 0
    },
    "segments": [
      {
        "seg_0_name": "Segment A",
        "seg_0_basrev": 1000,
        "seg_0_fyrs": 10,
        "seg_0_growth_mode": "Fade-Modell (g konvergiert zum Terminal-Wachstum)",
        "seg_0_fade_speed": 0.5,
        "seg_0_param_fade": true,
        "seg_0_tv_method": "Gordon Growth Model (Ewige Rente)",
        "seg_0_intra_corr": false,
        "s0_rg_dtype": "PERT-Verteilung", "s0_rg_pert_lo": 3.0, "s0_rg_pert_mo": 5.0, "s0_rg_pert_hi": 8.0,
        "s0_em_dtype": "PERT-Verteilung", "s0_em_pert_lo": 15.0, "s0_em_pert_mo": 18.0, "s0_em_pert_hi": 21.0,
        "s0_em_term_dtype": "Fest (Deterministisch)", "s0_em_term_fixed": 18.0,
        "s0_da_dtype": "Fest (Deterministisch)", "s0_da_fixed": 4.0,
        "s0_da_term_dtype": "Fest (Deterministisch)", "s0_da_term_fixed": 4.0,
        "s0_tx_dtype": "Fest (Deterministisch)", "s0_tx_fixed": 27.0,
        "s0_tx_term_dtype": "Fest (Deterministisch)", "s0_tx_term_fixed": 27.0,
        "s0_cx_dtype": "PERT-Verteilung", "s0_cx_pert_lo": 4.0, "s0_cx_pert_mo": 5.0, "s0_cx_pert_hi": 6.5,
        "s0_cx_term_dtype": "Fest (Deterministisch)", "s0_cx_term_fixed": 4.5,
        "s0_nwc_dtype": "Fest (Deterministisch)", "s0_nwc_fixed": 10.0,
        "s0_nwc_term_dtype": "Fest (Deterministisch)", "s0_nwc_term_fixed": 10.0,
        "s0_wacc_dtype": "Normalverteilung", "s0_wacc_n_mu": 8.5, "s0_wacc_n_sig": 0.5,
        "s0_tvg_dtype": "PERT-Verteilung", "s0_tvg_pert_lo": 1.5, "s0_tvg_pert_mo": 2.0, "s0_tvg_pert_hi": 2.5
      }
    ],
    "correlation": {}
  },
  "quellen": {
    "seg_0_basrev": "LTM: GJ 2024 + 9M25 − 9M24 [Q1]",
    "seg_0_fyrs": "Zyklus bis Normalisierung ca. 10 Jahre [Q2]",
    "s0_rg": "CAGR 2020–25 5,8 % [Q1]; Konsens 2026–28 6–7 % [Q2]",
    "bridge_nd": "Nettofinanzverschuldung 31.12.2025 [Q1]"
  },
  "quellenverzeichnis": {
    "Q1": "[UNTERNEHMEN], Q3-Bericht 2025, URL, abgerufen 2026-01-15 [primär geprüft]",
    "Q2": "LSEG Konsensus, Stand 2026-01-10 [sekundär]"
  }
}
```

(Die Vorlage zeigt nur einige Quellen-Einträge – in deiner Antwort ist
jeder Pflicht-Eintrag vorhanden.)

═══════════════════════════════════════════════════════════════════════════
SCHRITT 7 · PLAUSIBILITÄTS- & QUALITÄTSCHECKS (11 Checks)
═══════════════════════════════════════════════════════════════════════════

Führe ALLE 11 Checks durch. Dokumentiere das Ergebnis explizit:
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
Typische Bandbreiten:
  Blue-Chip: 7–10 % | Wachstum: 9–13 % | Emerging Markets: 11–16 %

────────────────────────────────────────────────────────────────────────
Check 4 · TV-Growth ≪ WACC
────────────────────────────────────────────────────────────────────────
Für JEDES Gordon-Growth-Segment: g < WACC − 3 pp?
Falls nicht → g senken.
Warum: TV = FCFF × (1+g) / (WACC − g). Bei g nahe WACC → TV → ∞.

────────────────────────────────────────────────────────────────────────
Check 5 · TV/EV-Ratio (Vorab-Schätzung)
────────────────────────────────────────────────────────────────────────
Schätze grob PV(TV) / EV je Segment:
  < 60 %  → ✅  |  60–75 % → ⚠️  |  > 75 % → ❌

Bei > 75 %: Prognosezeitraum verlängern oder FCFF-Margen prüfen.
Hoher TV-Anteil = Wert hängt fast nur vom Terminal Value ab →
geringe Prognosesicherheit.

────────────────────────────────────────────────────────────────────────
Check 6 · Implied ROIC
────────────────────────────────────────────────────────────────────────
Prüfe den ROIC mit der EXAKTEN Modellformel:

  NOPAT-Marge = (EBITDA% − D&A%) × (1 − Steuersatz)
  Reinvest-Marge = CAPEX% − D&A% + NWC% × g / (1 + g)
  ROIC = g × NOPAT-Marge / Reinvest-Marge

⚠️ NICHT die Lehrbuchformel „ROIC = NOPAT / Invested Capital" nehmen!
   Das Modell leitet ROIC aus der Steady-State-Identität g = ROIC × b ab,
   wobei b = Reinvest-Marge / NOPAT-Marge (Reinvestitionsquote).

Rechne das ROIC für deine geschätzten Parameter einmal durch:

  Beispiel: EBITDA% = 20 %, D&A% = 3 %, Steuer = 25 %,
            CAPEX% = 5 %, NWC% = 10 %, g = 5 %
  → NOPAT-Marge = (0,20 − 0,03) × (1 − 0,25) = 0,1275
  → Reinvest-Marge = 0,05 − 0,03 + 0,10 × 0,05/1,05 = 0,0248
  → ROIC = 0,05 × 0,1275 / 0,0248 = 25,7 %

Plausibel für die Branche?
  Software 25–50 % | Industrie 10–20 % | Handel 8–15 %
  Telko 8–15 % | Energie 10–18 % | Gesundheit 15–30 %

Ein ROIC > 100 % oder < 0 % signalisiert fast immer einen Parameterfehler!
AUSNAHMEN (Formelartefakt, kein Parameterfehler – im Kommentar so nennen):
  • NWC% negativ (Vorauszahlungen / Deferred Revenue) und CAPEX ≈ D&A:
    Reinvest-Marge → 0 oder negativ, ROIC explodiert oder kippt.
  • g ≤ 0 (schrumpfendes Segment): ROIC wird negativ, P(ROIC<WACC) ≈ 100 %.

────────────────────────────────────────────────────────────────────────
Check 7 · Fade-Konsistenz
────────────────────────────────────────────────────────────────────────
Konvergieren alle Terminal-Werte zu branchenüblichen Niveaus?
Ist die Fade-Richtung ökonomisch sinnvoll?

Prüfe insbesondere:
  • CAPEX-Terminal ≈ D&A-Terminal? (Im Steady State: Maintenance ≈ D&A)
  • EBITDA-Marge-Terminal ≤ Peer-Median + 5 pp?
  • Alle Terminal-Werte realistisch für den Steady State?

────────────────────────────────────────────────────────────────────────
Check 8 · Korrelationslogik
────────────────────────────────────────────────────────────────────────
Cross-Segment: Gleicher Endmarkt → höhere Korrelation. Matrix konsistent?
Intra-Segment: Vorzeichenlogik prüfen:
  • Wachstum ↔ CAPEX > 0?  (Wachstum erfordert Investition)
  • D&A ↔ CAPEX > 0?  (mehr Investition → mehr Abschreibung)
  • WACC ↔ Marge < 0?  (Risikoreichere Segmente haben niedrigere Margen)

────────────────────────────────────────────────────────────────────────
Check 9 · Impliziter Fair Value (Überschlagsrechnung)
────────────────────────────────────────────────────────────────────────
Berechne einen groben Punkt-DCF mit den Base-Case-Werten:

  1. FCFF (Punkt):
     Revenue_T = Base × (1 + g)^T
     EBITDA = Rev_T × EBITDA%
     NOPAT = (EBITDA − Rev_T × D&A%) × (1 − Steuer%)
     FCFF ≈ NOPAT − Rev_T × (CAPEX% − D&A%) + ...

  2. TV (Gordon oder Multiple)
  3. EV ≈ Σ Segment-EV − PV(Holding) − NetDebt − Minority − Pension
         + NonOp + Associates
  4. Kurs = EV / Aktien

Vergleiche mit aktuellem Aktienkurs: Premium/Discount in %.
> 100 % Abweichung → Parameterfehler wahrscheinlich → nochmal prüfen!

────────────────────────────────────────────────────────────────────────
Check 10 · Steady-State-Konsistenz
────────────────────────────────────────────────────────────────────────
Der Endwert rechnet mit den Terminal-Werten (siehe A). Prüfe je Segment:
  • D&A-Terminal ≈ 0,8–1,0 × CAPEX-Terminal?
  • CAPEX-Terminal gegen die Quote VOR einer Investitionsphase begründet?
  • Terminal-FCFF-Marge = (EBITDA − D&A)·(1 − t) + D&A − CAPEX − NWC·g/(1+g)
    plausibel gegen Historie und Peers? Wert im Kommentar nennen.

────────────────────────────────────────────────────────────────────────
Check 11 · Jahr-1-FCFF gegen Ist
────────────────────────────────────────────────────────────────────────
Rechne den Modell-FCFF für Jahr 1 (Summe der Segmente − Holdingkosten)
und vergleiche ihn mit dem LTM-Free-Cash-Flow bzw. der FCF-Guidance.
Zeige die Überleitung: SBC (im Modell Aufwand), Zinserträge (in der
Bridge über das Cash), M&A (falls als CAPEX modelliert), CAPEX-Anstieg.
Unerklärte Abweichung > 25 % → ⚠️ und Parameter prüfen.

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
| 10 | Steady-State-Konsistenz | ✅/⚠️/❌ | … |
| 11 | Jahr-1-FCFF gegen Ist | ✅/⚠️/❌ | … |


═══════════════════════════════════════════════════════════════════════════
SCHRITT 7b · PFLICHT-SZENARIEN (Strukturfragen)
═══════════════════════════════════════════════════════════════════════════

Die MC-Streuung erfasst nur Parameterunsicherheit. Strukturfragen
gehören in benannte Szenarien. Liefere MINDESTENS diese drei, jeweils
als Liste der geänderten JSON-Keys mit neuen Werten:

  1. SBC nicht als Aufwand: EBITDA-Margen (initial + terminal) um
     SBC/Umsatz erhöhen.
  2. WACC −1 pp und +1 pp (alle Segmente + Holding-Diskontsatz).
  3. Die größte Strukturfrage der Firma (z. B. dauerhafter vs. abebbender
     CAPEX, Erfolg einer Produktmigration, Normalisierung eines Zyklus),
     mit Begründung, warum gerade diese.

| Szenario | Geänderte Keys → Werte | Begründung |
|---|---|---|
| … | … | … |


═══════════════════════════════════════════════════════════════════════════
SCHRITT 8 · QUELLEN & DATENBASIS
═══════════════════════════════════════════════════════════════════════════

ZITIERFORMAT (VERPFLICHTEND):
[Q1] Autor/Institution, Titel, Quelle/Publisher, Datum, URL, Abrufdatum
[Q2] Autor/Institution, Titel, Quelle/Publisher, Datum, URL, Abrufdatum
…
Verwende in allen Begründungen ausschließlich Kurzverweise wie [Q1], [Q2].
Jede Quelle trägt ihren Prüfstatus:
  [primär geprüft]     – Zahl im Originaldokument (Bericht, Filing) gelesen
  [primär, teilweise]  – Original gelesen, aber nicht vollständig
  [sekundär]           – nur über Presse, Transkript, Datenportal
Zentrale Zahlen (Umsätze, Bridge) sollen [primär geprüft] sein.

| Kürzel | Typ | Vollzitat |
|---|---|---|
| Q1 | Geschäftsbericht | [UNTERNEHMEN] Annual Report [GJ], URL, Abrufdatum |
| Q2 | Analystenberichte | [Bloomberg Consensus / LSEG / …], URL/Terminal-Hinweis |
| Q3 | Branchenreports | [Gartner / IDC / McKinsey / Statista / …], URL |
| Q4 | Damodaran | [Betas, ERP, Multiples – pages.stern.nyu.edu/~adamodar/], URL |
| Q5 | Marktdaten | [Anleiherenditen, CDS, Ratings], Quelle + Datum |
| Q6 | Sonstige | … |


═══════════════════════════════════════════════════════════════════════════
ZUSAMMENFASSUNG (EXECUTIVE SUMMARY)
═══════════════════════════════════════════════════════════════════════════

Danach eine Executive Summary (max. 5 Sätze), ganz am Schluss folgt der
JSON-Block aus Schritt 6:
  1. Berichtswährung + Anzahl Segmente
  2. Wichtigste Werttreiber-Unsicherheiten (die breitesten Verteilungen)
  3. Welche Segmente treiben den Wert, welche sind marginal?
  4. Erwartete Bewertungsrichtung vs. aktueller Kurs
  5. Empfehlung zur Interpretation der MC-Ergebnisse
     (auf welche Quantile achten, wo liegt das Risiko?)
````

---

## Beispiel-Anwendung

Ersetze `[UNTERNEHMEN]` und führe den Prompt aus:

```
[UNTERNEHMEN] = Siemens AG
```
