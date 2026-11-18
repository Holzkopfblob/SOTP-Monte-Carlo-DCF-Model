"""
Transaktionsbuch: Bestand, Einstand und realisierte Gewinne aus Käufen, Verkäufen, Dividenden und Gebühren.

Steuerlich wie die Bank (§ 20 Abs. 4 EStG): Veräußerungsgewinn = Erlös − Veräußerungskosten − Anschaffungskosten
inkl. Nebenkosten, verbrauchte Stücke nach FIFO (zuerst gekaufte zuerst verkauft). Bruchstücke erlaubt.

Transaktion (dict): datum (JJJJ-MM-TT), ticker, typ ∈ {kauf, verkauf, dividende, gebuehr}, stueck, kurs_eur
(je Stück), betrag_eur (Dividende netto gutgeschrieben bzw. Gebühr ohne Order), steuer_eur, notiz.
Bei Trade Republic ist die Ordergebühr im Kurs enthalten (Kauf: Einstand inkl. Gebühr, Verkauf: Erlös nach Gebühr,
Nutzerentscheid 18.11.2026); gebuehr_eur bleibt optional für andere Banken.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TYPEN = ("kauf", "verkauf", "dividende", "gebuehr")
EPS = 1e-9


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v          # NaN ⇒ 0


@dataclass
class Konto:
    """Ergebnis je Ticker."""
    stueck: float = 0.0
    einstand_gesamt: float = 0.0          # Anschaffungskosten der offenen Stücke inkl. Kaufgebühren
    erster_kauf_offen: str = ""           # Kaufdatum des ältesten noch offenen Lots
    realisiert: float = 0.0               # Veräußerungsgewinne (nach Gebühren), FIFO
    dividenden: float = 0.0               # netto gutgeschrieben
    gebuehren: float = 0.0                # alle Gebühren (Kauf, Verkauf, sonstige)
    steuern: float = 0.0                  # einbehaltene Steuern (Quellensteuer, KapESt)
    investiert: float = 0.0               # Summe aller Kaufbeträge inkl. Gebühren
    erloese: float = 0.0                  # Summe aller Verkaufserlöse nach Gebühren
    lots: list = field(default_factory=list)   # [datum, stueck, kosten_je_stueck]
    warnungen: list = field(default_factory=list)

    @property
    def einstand(self) -> float:
        """Durchschnittlicher Einstand je offenem Stück (inkl. Kaufgebühren)."""
        return self.einstand_gesamt / self.stueck if self.stueck > EPS else float("nan")

    @property
    def geschlossen(self) -> bool:
        return self.stueck <= EPS and self.investiert > 0


def book(transaktionen: list[dict]) -> dict[str, Konto]:
    """Bucht alle Transaktionen chronologisch (stabil: gleiche Datumsreihenfolge wie erfasst)."""
    out: dict[str, Konto] = {}
    for t in sorted(transaktionen, key=lambda t: str(t.get("datum") or "")):
        k = out.setdefault(str(t["ticker"]), Konto())
        typ = str(t.get("typ") or "").lower()
        st, kurs, fee = _f(t.get("stueck")), _f(t.get("kurs_eur")), _f(t.get("gebuehr_eur"))
        k.gebuehren += fee
        k.steuern += _f(t.get("steuer_eur"))
        if typ == "kauf":
            if st <= 0:
                k.warnungen.append(f"{t.get('datum')}: Kauf ohne Stückzahl")
                continue
            kosten = st * kurs + fee
            k.lots.append([str(t.get("datum") or ""), st, kosten / st])
            k.investiert += kosten
        elif typ == "verkauf":
            erloes = st * kurs - fee - _f(t.get("steuer_eur"))
            k.erloese += erloes
            rest, kosten = st, 0.0
            while rest > EPS and k.lots:
                lot = k.lots[0]
                take = min(rest, lot[1])
                kosten += take * lot[2]
                lot[1] -= take
                rest -= take
                if lot[1] <= EPS:
                    k.lots.pop(0)
            if rest > 1e-6:
                k.warnungen.append(f"{t.get('datum')}: Verkauf von {rest:.6f} Stück mehr als im Bestand")
            k.realisiert += st * kurs - fee - kosten     # Steuer mindert den Gewinn nicht (wird separat geführt)
        elif typ == "dividende":
            k.dividenden += _f(t.get("betrag_eur"))
        elif typ == "gebuehr":
            k.gebuehren += _f(t.get("betrag_eur"))
        else:
            k.warnungen.append(f"{t.get('datum')}: unbekannter Typ „{typ}“")
    for k in out.values():
        k.stueck = sum(lot[1] for lot in k.lots)
        k.einstand_gesamt = sum(lot[1] * lot[2] for lot in k.lots)
        k.erster_kauf_offen = k.lots[0][0] if k.lots else ""
    return out


def apply_to_positions(positions: list[dict], transaktionen: list[dict]) -> tuple[list[dict], dict[str, Konto]]:
    """Positionen mit Stück, Einstand und Kaufdatum aus dem Buch überschreiben (nur Ticker mit Transaktionen);
    Ticker im Buch ohne Position werden als Position ergänzt. Rückgabe: (Positionen, Konten)."""
    konten = book(transaktionen)
    out = []
    for pos in positions:
        k = konten.get(pos["ticker"])
        if k is not None:
            pos = {**pos, "stueck": round(k.stueck, 6),
                   "einstand_eur": round(k.einstand, 4) if k.stueck > EPS else None,
                   "kaufdatum": k.erster_kauf_offen}
        out.append(pos)
    return out, konten
