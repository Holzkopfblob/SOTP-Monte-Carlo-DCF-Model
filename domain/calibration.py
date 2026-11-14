"""
Kalibrierung: Wie treffsicher sind die Verteilungen des Modells?

PIT (Probability Integral Transform): Für jede Prognose der Quantil-Rang des später realisierten
Werts in der damaligen Modellverteilung. Bei guter Kalibrierung sind die Ränge über viele Prognosen
gleichverteilt auf [0, 1]: ~10 % je Dezil, ~90 % im P5–P95-Band. Häufung unten ⇒ Modell zu
optimistisch, Häufung an beiden Rändern ⇒ Bänder zu eng, Häufung in der Mitte ⇒ zu breit.
"""
from __future__ import annotations

import numpy as np


def pit(samples: np.ndarray, actual: float) -> float:
    """Quantil-Rang von *actual* in *samples* (Bindungen zur Hälfte)."""
    s = np.asarray(samples, float)
    return float((np.sum(s < actual) + 0.5 * np.sum(s == actual)) / len(s))


def year1_growth(g0: np.ndarray, g_term: np.ndarray | float, fade_speed: float, fade: bool) -> np.ndarray:
    """Umsatzwachstum in Jahr 1 wie im Modell: g1 = g_term + (g0 − g_term)·e^(−λ); konstant ⇒ g0."""
    g0 = np.asarray(g0, float)
    return g_term + (g0 - g_term) * np.exp(-fade_speed) if fade else g0


def aggregate(base_rev: list[float], growth: list[np.ndarray], margin: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Konzernwerte je Szenario: Umsatzwachstum und umsatzgewichtete Marge in Jahr 1."""
    rev1 = [b * (1 + g) for b, g in zip(base_rev, growth)]
    total1 = np.sum(rev1, axis=0)
    g = total1 / float(sum(base_rev)) - 1
    m = np.sum([r * mg for r, mg in zip(rev1, margin)], axis=0) / total1
    return g, m


def summarize(pits: list[float]) -> dict:
    """Kennzahlen eines PIT-Satzes (Anteile in [0, 1])."""
    p = np.asarray(pits, float)
    if len(p) == 0:
        return {"n": 0}
    counts, _ = np.histogram(p, bins=10, range=(0, 1))
    return {
        "n": int(len(p)),
        "mittel": float(p.mean()),                      # 0,5 ideal; < 0,5 ⇒ Ist liegt eher unter dem Modell
        "im_90_band": float(np.mean((p > 0.05) & (p < 0.95))),
        "unter_p10": float(np.mean(p < 0.10)),
        "ueber_p90": float(np.mean(p > 0.90)),
        "dezile": counts.tolist(),
    }


def verdict(s: dict) -> str:
    """Einfache Lesart (bei kleinem n nur Tendenz)."""
    if s.get("n", 0) < 5:
        return "zu wenige Prüfpunkte"
    notes = []
    if s["mittel"] < 0.35:
        notes.append("Modell zu optimistisch")
    elif s["mittel"] > 0.65:
        notes.append("Modell zu pessimistisch")
    if s["im_90_band"] < 0.75:
        notes.append("Bänder zu eng")
    elif s["im_90_band"] > 0.98 and s["n"] >= 20:
        notes.append("Bänder evtl. zu breit")
    return ", ".join(notes) or "keine systematische Abweichung erkennbar"
