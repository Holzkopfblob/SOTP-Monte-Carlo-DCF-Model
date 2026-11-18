"""
Rückblick-Labor (Baustein 6, Plan: Vault 06_INVEST/SOTP-Modell/Plan Rückblick-Labor (Baustein 6).md).

Misst die Quoten des Modells: Eine schlanke Bewertung mit derselben Engine wird rückwirkend auf US-Firmen angewandt
(Stichtag 30.06. eines Jahres t, Zahlen des Geschäftsjahres t−1, Basisraten nur aus Jahren ≤ t−1) und mit dem
verglichen, was danach geschah.

Firmenebene statt Aktie: Preis = Börsenwert des Streubesitzes am 30.06. (dei:EntityPublicFloat, liegt auch für später
verschwundene Firmen vor), Aktienzahl 1. Ergebnis = Aktionärsrendite der Firma über h Jahre: Public Float t+h plus
Dividenden und Rückkäufe minus Kapitalerhöhungen. So sind Überlebende und Abgänge gleich gemessen.

Entscheidungsgröße wie im Portfolio (Baustein 1): Eigenkapitalrendite zum Kurs r_E = r + D/E·(r − k_d) gegen die
ETF-Rendite damals (rf 3 J. + implizite ERP von Damodaran zum Jahresende t−1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from application.research_kit import FEST, PERT, build_config, d, fade_pert, segment
from domain.base_rates import G_EDGES, M_EDGES, S_EDGES, bucket, conditional, outcomes, sektor
from domain.irr import R_MAX, equity_at, implied_irr
from infrastructure.config_io import build_simulation_config
from infrastructure.monte_carlo_engine import MonteCarloEngine

# Damodaran, histimpl.html (Jan. 2026), Spalte „Implied ERP (FCFE)“, Jahresende [primär, per Abruf 18.11.2026]
ERP_JAHRESENDE = {2011: 6.01, 2012: 5.78, 2013: 4.96, 2014: 5.78, 2015: 6.12, 2016: 5.69, 2017: 5.08, 2018: 5.96,
                  2019: 5.20, 2020: 4.72, 2021: 4.24, 2022: 5.94, 2023: 4.60, 2024: 4.33, 2025: 4.23}
KD_SPREAD, KD_TAX = 1.5, 0.25      # wie die Portfolio-Pauschale (rf + 1,5 %)·0,75
LAMBDA, G_TERM = 0.45, 2.5
H_BASIS = 3                        # Basisraten-Horizont (Jahre); 5 J. wäre vor 2015 nicht zeitpunktgenau verfügbar
CAGR_Q = np.arange(5, 100, 5)      # gespeicherte Quantile der Modell-Umsatz-CAGR (5 J.) für die Kalibrierung


def faelle(panel: pd.DataFrame, lab: pd.DataFrame, t: int, min_umsatz: float = 5e8) -> pd.DataFrame:
    """Fälle zum Stichtag 30.06.t: Geschäftsjahr t−1 (Umsatz ≥ *min_umsatz*), Public Float am 30.06.t, ohne Finanzwerte."""
    b = panel[panel.jahr == t - 1][["cik", "name", "sic", "umsatz", "ebit", "da", "capex"]]
    alt = lab[lab.jahr == t - 1].set_index("cik").get("capex_alt")
    if alt is not None:                  # Capex unter anderen Tags (nur im Labor, Basisraten der App unberührt)
        b = b.assign(capex=b.capex.fillna(b.cik.map(alt)))
    vj = panel[panel.jahr == t - 2][["cik", "umsatz"]].rename(columns={"umsatz": "umsatz_vj"})
    nd = lab[lab.jahr == t - 1][["cik", "nettoschulden"]]
    pf = lab[lab.jahr == t][["cik", "public_float"]]
    f = b.merge(vj, on="cik", how="left").merge(nd, on="cik", how="left").merge(pf, on="cik", how="inner")
    f["sektor"] = f.sic.map(sektor)
    f = f[(f.umsatz >= min_umsatz) & (f.sektor != "Finanz") & (f.public_float > 0)
          & f[["ebit", "da", "capex"]].notna().all(axis=1)].copy()
    f["marge"] = (f.ebit + f.da) / f.umsatz
    f["g_vj"] = f.umsatz / f.umsatz_vj - 1
    # ohne Schuldenangabe kein Fall: schuldenfrei rechnen hieße, verschuldete Firmen als Schnäppchen zu zeigen
    f = f[f.nettoschulden.notna()]
    return f[(f.marge > -0.5) & (f.marge < 0.9)].reset_index(drop=True)


def basisraten(panel: pd.DataFrame, t: int) -> pd.DataFrame:
    """Ergebnisse über H_BASIS Jahre, die am Stichtag t schon bekannt waren (Ende ≤ t−1)."""
    return outcomes(panel[panel.jahr <= t - 1], H_BASIS)


def light_config(fall: pd.Series, ob: pd.DataFrame, t: int, n_sim: int = 2000) -> dict:
    """Ein-Segment-Bewertung (Mio. USD) aus dem 10-K t−1 und den Basisraten; Preis = Public Float, Aktien = 1."""
    bed = dict(sektor=fall.sektor, groesse=int(bucket([fall.umsatz], S_EDGES)[0]),
               g_start=int(bucket([fall.g_vj], G_EDGES)[0]) if np.isfinite(fall.g_vj) else None,
               m_start=int(bucket([fall.marge], M_EDGES)[0]))
    g, _ = conditional(ob, "cagr", **bed)
    dm, _ = conditional(ob, "dm", **bed)
    g5, g50, g95 = np.percentile(g, [5, 50, 95]) * 100
    d5, d50, d95 = np.percentile(dm, [5, 50, 95]) * 100
    m0 = fall.marge * 100
    da = max(fall.da / fall.umsatz * 100, 0.1)
    cx = max(fall.capex / fall.umsatz * 100, 0.1)
    # ponytail: Endwert-Capex grob = D&A + halbe Lücke (mind. D&A·1,1, sonst wächst der Endwert ohne Investition)
    cx_t = max(da * 1.1, da + 0.5 * (cx - da))
    tax = 30.0 if t <= 2017 else 24.0          # vor/nach US-Steuerreform 2018, Bund + Staaten effektiv
    em_t = sorted([m0 + d5, m0 + d50, m0 + d95])
    seg = segment(0, "Firma", fall.umsatz / 1e6, 10, LAMBDA,
                  fade_pert(g50, G_TERM, LAMBDA, max((g95 - g5) / 2, 1.0)), (FEST, round(m0, 2)),
                  (PERT, round(em_t[0], 2), round(em_t[1], 2), round(em_t[2] + 1e-3, 2)),
                  (FEST, round(da, 2)), (FEST, round(da, 2)), (FEST, tax), (FEST, tax),
                  (FEST, round(cx, 2)), (FEST, round(cx_t, 2)), (FEST, 10.0), (FEST, 10.0),
                  (FEST, 8.0), (PERT, 1.5, G_TERM, 3.0))
    bridge = {**d("bridge_cc", FEST, 0.0), **d("bridge_cd", FEST, 8.0), **d("bridge_nd", FEST, fall.nettoschulden / 1e6),
              **d("bridge_sh", FEST, 1.0), **d("bridge_mi", FEST, 0.0), **d("bridge_pn", FEST, 0.0),
              **d("bridge_no", FEST, 0.0), **d("bridge_as", FEST, 0.0), "setup_ext_bridge": True}
    return build_config(company=str(fall.cik), ccy="USD", kurs=fall.public_float / 1e6, stichtag=f"{t}-06-30",
                        stand="Labor", basis=f"FY{t - 1}", bilanz=f"{t - 1}-12-31", segments=[seg], bridge=bridge,
                        corr={}, quellen={}, qv={}, szenarien=[], kd=(KD_SPREAD, KD_TAX * 100), n_sim=n_sim)


def bewerte(fall: pd.Series, ob: pd.DataFrame, t: int, rf: float, n_sim: int = 2000) -> dict:
    """Engine-Lauf und Kennzahlen eines Falls (Anteile, nicht %). *rf*: US-Treasury 3 J. am Stichtag."""
    cfg = light_config(fall, ob, t, n_sim)
    res = MonteCarloEngine(build_simulation_config(cfg)).run()
    preis = fall.public_float / 1e6
    r = implied_irr(res, preis)
    d_e = fall.nettoschulden / fall.public_float
    kd = (rf + KD_SPREAD / 100) * (1 - KD_TAX)
    r_e = np.maximum(r + d_e * (r - kd), -0.99)
    etf = rf + ERP_JAHRESENDE[t - 1] / 100
    wert = equity_at(res, np.full(len(r), etf))
    det = next(iter(res.segment_cashflows.values()))
    cagr5 = (det.revenue[:, 4] / det.base_revenue) ** (1 / 5) - 1
    return {"cik": int(fall.cik), "t": t, "name": fall["name"], "sektor": fall.sektor, "umsatz": fall.umsatz,
            "public_float": fall.public_float, "nettoschulden": fall.nettoschulden, "rf": rf, "etf": etf,
            **{f"irr_p{q}": float(np.percentile(r, q)) for q in (5, 25, 50, 75, 95)},
            **{f"re_p{q}": float(np.percentile(r_e, q)) for q in (5, 25, 50, 75, 95)},
            "vorsprung": float(np.median(r_e) - etf), "wert_kurs_p50": float(np.median(wert) / preis),
            "am_rand": float(np.mean(r >= R_MAX - 1e-6)),
            "cagr5_q": np.percentile(cagr5, CAGR_Q).round(5).tolist()}


def ergebnisse(panel: pd.DataFrame, lab: pd.DataFrame, t: int, h: int = 5) -> pd.DataFrame:
    """Tatsächliche Aktionärsrendite je Firma von 30.06.t bis 30.06.t+h (bzw. bis zum letzten Public Float):
    (Float_Ende + Σ Dividenden + Rückkäufe − Emissionen) / Float_t. Dazu Umsatz-CAGR FY t−1 → t−1+h.
    ponytail: Ausschüttungen der Kalenderjahre t … t+h−1 ohne Wiederanlage (Index mit Wiederanlage ⇒ leicht konservativ)."""
    pf = lab.pivot_table(index="cik", columns="jahr", values="public_float")
    pay = lab.assign(netto=lab.dividenden.fillna(0) + lab.rueckkauf.fillna(0) - lab.emission.fillna(0)) \
        .pivot_table(index="cik", columns="jahr", values="netto").fillna(0.0)
    rev = panel.pivot_table(index="cik", columns="jahr", values="umsatz")
    out = []
    for cik in pf.index[pf.get(t, pd.Series(dtype=float)).reindex(pf.index).notna()]:
        jahre = [y for y in range(t + 1, t + h + 1) if y in pf.columns and np.isfinite(pf.at[cik, y])]
        ende = max(jahre) if jahre else t
        aus = sum(pay.at[cik, y] for y in range(t, ende) if cik in pay.index and y in pay.columns)
        rv0, rvh = (rev.at[cik, y] if cik in rev.index and y in rev.columns else np.nan for y in (t - 1, t - 1 + h))
        out.append({"cik": int(cik), "t": t, "float_t": pf.at[cik, t], "float_ende": pf.at[cik, ende],
                    "ende_jahr": ende, "abgang": ende < t + h, "ausschuettung": aus,
                    "umsatz_cagr5": (rvh / rv0) ** (1 / h) - 1 if rv0 and rv0 > 0 and rvh and rvh > 0 else np.nan})
    o = pd.DataFrame(out)
    o["tsr_roh"] = (o.float_ende + o.ausschuettung) / o.float_t - 1
    return o
