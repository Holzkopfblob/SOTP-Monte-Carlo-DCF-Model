"""Rückblick-Labor: Auswertung (Signal, k, Kalibrierung, Schwellen, Fallen). Schreibt labor_faelle.parquet und druckt den
Bericht. Lernzeitraum 2014–2017 (Schwellen wählen), Prüfzeitraum 2018–2020 (einmal prüfen, nicht nachjustieren).

Aufruf:  python scripts/labor_auswertung.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application.labor import CAGR_Q, ergebnisse  # noqa: E402
from infrastructure.sec_xbrl import abgang_grund, build_lab_panel, data_dir  # noqa: E402

H = 5
LERN, PRUEF = range(2014, 2018), range(2018, 2021)
# Abgang: Faktor auf den letzten Börsenwert (Insolvenz = Totalverlust; unklar = Shumway 1997 −30 % bzw. 0 %)
VARIANTEN = {"vorsichtig": {"insolvenz": 0.0, "uebernahme": 1.0, "unbekannt": 0.7},
             "guenstig": {"insolvenz": 0.0, "uebernahme": 1.3, "unbekannt": 1.0}}


def index_pa() -> dict[int, float]:
    """S&P 500 Total Return p. a. vom 30.06.t bis 30.06.t+H."""
    import yfinance as yf
    s = yf.download("^SP500TR", start="2014-01-01", progress=False, auto_adjust=True)["Close"].squeeze()
    at = lambda y: float(s[:f"{y}-06-30"].iloc[-1])         # noqa: E731
    return {t: (at(t + H) / at(t)) ** (1 / H) - 1 for t in range(2014, 2021)}


def faelle_mit_ergebnis() -> pd.DataFrame:
    b = pd.read_parquet(data_dir() / "labor_bewertungen.parquet")
    b = b[b.get("fehler").isna()] if "fehler" in b else b
    panel = pd.read_parquet(data_dir() / "panel.parquet")
    lab = build_lab_panel()
    o = pd.concat([ergebnisse(panel, lab, t, H) for t in sorted(b.t.unique())], ignore_index=True)
    d = b.merge(o, on=["cik", "t"], how="inner")
    grund = abgang_grund(d.loc[d.abgang, "cik"].unique())
    d["grund"] = np.where(d.abgang, d.cik.map(grund).fillna("unbekannt"), "")
    idx = index_pa()
    d["index_pa"] = d.t.map(idx)
    for v, mult in VARIANTEN.items():
        m = np.where(d.abgang, d.grund.map(mult).fillna(mult["unbekannt"]), 1.0)
        tsr = (d.float_ende * m + d.ausschuettung) / d.float_t - 1
        d[f"tsr_pa_{v}"] = np.maximum(1 + tsr, 0) ** (1 / H) - 1
        d[f"ueber_{v}"] = d[f"tsr_pa_{v}"] - d.index_pa
    d["d_e"] = d.nettoschulden / d.public_float
    return d


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip(x, lo, hi)
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    # HC1-robuste Standardfehler
    xtx = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx @ (X.T * e ** 2) @ X @ xtx) * len(x) / (len(x) - 2))
    return b, se, len(x)


def bericht(d: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print(f"Fälle {len(d)} (davon Abgänge {d.abgang.mean():.0%}: "
          + ", ".join(f"{k} {v}" for k, v in d[d.abgang].grund.value_counts().items()) + ")")
    print("Index p. a. je Jahrgang:", {t: f"{v:.1%}" for t, v in d.groupby("t").index_pa.first().items()})
    d = d[d.am_rand <= 0.05]
    for v in VARIANTEN:
        u = f"ueber_{v}"
        print(f"\n=== Variante {v} ===")
        for name, jahre in (("Lernen 2014–17", LERN), ("Prüfen 2018–20", PRUEF)):
            x = d[d.t.isin(jahre)].copy()
            x["zehntel"] = x.groupby("t").vorsprung.transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
            g = x.groupby("zehntel")[u].agg(["mean", "median", lambda s: (s > 0).mean(), "count"])
            g.columns = ["Überrendite Mittel", "Median", "Trefferquote", "n"]
            print(f"\n{name}: Überrendite p. a. gegen S&P 500 TR nach Zehntel des Vorsprungs (9 = höchster)")
            print(g.round(3).to_string())
            b, se, n = ols(x.vorsprung, x[u])
            print(f"k (Überrendite auf Vorsprung): {b[1]:.2f} ± {1.645 * se[1]:.2f} (90 %), Achse {b[0]:+.3f}, n {n}")
            b2, se2, _ = ols(x.re_p50, x[f"tsr_pa_{v}"])
            print(f"Rendite tatsächlich auf Rendite zum Kurs (P50): Steigung {b2[1]:.2f} ± {1.645 * se2[1]:.2f}, "
                  f"Achse {b2[0]:+.3f}")
            print("Schwellen (Vorsprung > m): " + "; ".join(
                f"m {m:.0%}: n {int((x.vorsprung > m).sum())}, Ü {x.loc[x.vorsprung > m, u].mean():+.2%}, "
                f"Treffer {(x.loc[x.vorsprung > m, u] > 0).mean():.0%}" for m in (0, .02, .04, .06, .08)))
    # Kalibrierung (variantenunabhängig für Umsatz; Rendite vorsichtig)
    q = np.array(CAGR_Q) / 100
    pit = [np.interp(r, np.asarray(c), q, left=0.0, right=1.0)
           for r, c in zip(d.umsatz_cagr5, d.cagr5_q) if np.isfinite(r)]
    pit = np.array(pit)
    print(f"\n=== Kalibrierung ===\nUmsatz-CAGR 5 J.: im Band P5–P95 {np.mean((pit > .05) & (pit < .95)):.0%} (Soll 90 %), "
          f"unter P5 {np.mean(pit <= .05):.0%}, über P95 {np.mean(pit >= .95):.0%}, Median-PIT {np.median(pit):.2f} (Soll 0,50)")
    r = d.tsr_pa_vorsichtig
    print(f"Aktionärsrendite p. a. im Band re_P5–re_P95: {np.mean((r > d.re_p5) & (r < d.re_p95)):.0%} (Soll 90 %), "
          f"unter P5 {np.mean(r <= d.re_p5):.0%}, über P95 {np.mean(r >= d.re_p95):.0%}")
    # Fallen: oberstes Fünftel nach Hebel und Sektor (vorsichtig, alle Jahre)
    d = d.assign(fuenftel=d.groupby("t").vorsprung.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop")))
    top = d[d.fuenftel == 4]
    top = top.assign(hebel=pd.cut(top.d_e, [-np.inf, 0, 0.5, 1.0, np.inf], labels=["Netto-Kasse", "≤0,5", "0,5–1", ">1"]))
    print("\n=== Fallen: oberstes Fünftel, Überrendite (vorsichtig) nach Hebel ===")
    print(top.groupby("hebel", observed=True).ueber_vorsichtig.agg(["mean", "median", "count"]).round(3).to_string())
    print("\nnach Sektor (n ≥ 30):")
    s = top.groupby("sektor").ueber_vorsichtig.agg(["mean", "median", "count"])
    print(s[s["count"] >= 30].sort_values("mean").round(3).to_string())


if __name__ == "__main__":
    d = faelle_mit_ergebnis()
    d.drop(columns=["cagr5_q"]).to_parquet(data_dir() / "labor_faelle.parquet")
    bericht(d)
