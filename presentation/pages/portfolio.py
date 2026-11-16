"""
Portfolio – Zielgewichte und Orders aus den SOTP-Bewertungen
=============================================================
Depot (Stück, Cash, Gruppe) liegt in einer JSON außerhalb des Repos; Pfad per Umgebungsvariable
``SOTP_DEPOT`` vorbelegt. Optimierung: Fractional Kelly (CRRA) auf der Szenario-Mischung
„Modell hat recht“ / „Markt hat recht“ (application.portfolio_service).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from application.portfolio_service import (
    Params, find_valuations, load_depot, rebalance, save_depot, simulate_valuation,
)
from infrastructure.market_data import fetch_bund_3y, fetch_closes
from presentation.charts.portfolio import allocation_chart, corr_heatmap, upside_chart, wealth_chart

_NEWEST = "(neueste)"
_CASH = "Cash (EUR)"


@st.cache_data(show_spinner=False)
def _simulate(path: str, mtime: float) -> dict:  # mtime nur als Cache-Schlüssel
    return simulate_valuation(path)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _bund_3y() -> tuple[float, str]:
    try:
        return fetch_bund_3y()
    except Exception:  # offline / API geändert → letzter geprüfter Wert
        return 0.0315, "Fallback 13.11.2026"


@st.cache_data(ttl=3600, show_spinner=False)
def _closes(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_closes(list(tickers))


def _eur(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(x: float, signed: bool = False) -> str:
    return (f"{x:+.1%}" if signed else f"{x:.1%}").replace(".", ",")


def _short(name: str) -> str:
    for s in (", Inc.", " Inc.", " AG", " Corporation", " Ltd.", " Group"):
        name = name.replace(s, "")
    return name


# ── Bereiche ──────────────────────────────────────────────────────────────────────────────

def _depot_tab(depot: dict, path: str, vals: dict, files: dict) -> tuple[list[dict], float, str]:
    st.markdown("##### Positionen")
    st.caption("Stück 0 = Beobachtung. Titel derselben **Gruppe** wechseln in der Szenario-Mischung gemeinsam "
               "in die Welt „Markt hat recht“ (gemeinsamer Modellfehler, z. B. alle Autobauer).")
    cols = ["firma", "ticker", "stueck", "einstand_eur", "kaufdatum", "gruppe", "datei"]
    df = pd.DataFrame(depot.get("positionen") or [], columns=cols)
    df["datei"] = df["datei"].fillna("").replace("", _NEWEST)
    df["gruppe"] = df["gruppe"].fillna("")
    edited = st.data_editor(
        df, num_rows="dynamic", width="stretch", key="pf_editor", hide_index=True,
        column_config={
            "firma": st.column_config.SelectboxColumn("Firma", options=sorted(vals), required=True, width="large"),
            "ticker": st.column_config.TextColumn("Ticker (Yahoo)", required=True),
            "stueck": st.column_config.NumberColumn("Stück", min_value=0.0, step=0.000001, format="%.6f", default=0.0),
            "einstand_eur": st.column_config.NumberColumn("Einstand €/Stück", min_value=0.0, format="%.2f"),
            "kaufdatum": st.column_config.TextColumn("Kaufdatum"),
            "gruppe": st.column_config.TextColumn("Gruppe", help="z. B. Auto; leer = eigene Gruppe"),
            "datei": st.column_config.SelectboxColumn("Bewertung", options=[_NEWEST, *sorted(files)], default=_NEWEST),
        },
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    cash = c1.number_input("Cash (EUR)", min_value=0.0, value=float(depot.get("cash_eur", 0.0)), step=100.0)
    folder = c3.text_input("Bewertungen-Ordner", depot.get("bewertungen_ordner", ""), key="pf_folder")
    positions = [
        {"firma": r.firma, "ticker": str(r.ticker).strip(), "stueck": float(r.stueck or 0),
         "datei": "" if r.datei in (None, _NEWEST) else r.datei,
         "einstand_eur": None if pd.isna(r.einstand_eur) else float(r.einstand_eur),
         "kaufdatum": r.kaufdatum if isinstance(r.kaufdatum, str) else "",
         "gruppe": r.gruppe.strip() if isinstance(r.gruppe, str) else ""}
        for r in edited.itertuples() if r.firma and isinstance(r.ticker, str) and r.ticker.strip()
    ]
    c2.markdown("&nbsp;")
    if c2.button("💾 Depot speichern", width="stretch"):
        save_depot(path, {"bewertungen_ordner": folder, "cash_eur": cash, "positionen": positions})
        st.session_state.pf_depot = load_depot(path)
        st.toast(f"Gespeichert: {path}", icon="✅")
    return positions, cash, folder


def _settings() -> Params:
    rf_default, rf_day = _bund_3y()
    with st.expander("⚙️ Annahmen der Optimierung", expanded=False):
        a, b, c = st.columns(3)
        p_market = a.slider("„Markt hat recht“ (p)", 0.0, 0.9, 0.3, 0.05,
                            help="Szenario-Mischung: In diesem Anteil der Szenarien ist der Kurs fair und die Aktie "
                                 "verdient nur ihre Marktrendite (Zins + Risikoprämie). 0 = Modell voll glauben.")
        fraction = b.slider("Kelly-Faktor", 0.1, 1.0, 0.5, 0.05, help="½ = halbes Kelly (CRRA γ = 2).")
        max_w = c.slider("Max. Gewicht je Titel (Notbremse)", 0.05, 1.0, 0.35, 0.01)
        d, e, f = st.columns(3)
        erp = d.number_input("Risikoprämie Markt-Welt p. a.", 0.0, 0.1, 0.05, 0.005, format="%.3f")
        rf = e.number_input("Cash-Zins p. a. (Bund 3 J.)", 0.0, 0.1, rf_default, 0.0025, format="%.4f",
                            help=f"Bundesbank-Zinsstruktur 3 Jahre, Stand {rf_day}.")
        min_order = f.number_input("Mindestorder (EUR)", 0.0, value=100.0, step=50.0)
        g, h, i = st.columns(3)
        min_cash = g.slider("Mindest-Cash", 0.0, 0.5, 0.0, 0.01, help="0 = Cash-Quote bestimmt der Optimizer.")
        allow_short = h.toggle("Short erlauben", False, help="Brutto Σ|Gewicht| ≤ 1 − Mindest-Cash (kein Hebel).")
        group_cap = i.slider("Max. Gewicht je Gruppe", 0.05, 1.0, 0.20, 0.01,
                             help="Summe aller Titel einer benannten Gruppe (Spalte „Gruppe“ im Depot), z. B. Auto.")
        factor_shrink = st.slider("Value-Stil herausnehmen (s)", 0.0, 1.0, 0.0, 0.05,
                                  help="Das Modell hält Firmen mit niedrigem EV/EBITDA systematisch für billiger "
                                       "(Querschnitt 16.11.2026). s = 0: Modell voll glauben, s = 1: diesen Anteil "
                                       "der Lücke ganz herausnehmen, nur die firmenspezifische Überzeugung bleibt.")
        rho_world = st.slider("Gemeinsamer Modellfehler (ρ)", 0.0, 1.0, 0.4, 0.05,
                              help="Wie stark „Markt hat recht“ alle Titel zugleich trifft. 0 = je Gruppe unabhängig, "
                                   "1 = ein Schalter für alle. 0,4: Die Lücke Modell/Markt hängt zu ~40 % am "
                                   "Marktmultiple (Value-Stil des Modells, Querschnitt 16.11.2026).")
    return Params(fraction=fraction, max_w=max_w, min_cash=min_cash, min_order_eur=min_order,
                  allow_short=allow_short, rf=rf, p_market=p_market, erp=erp, group_cap=group_cap,
                  rho_world=rho_world, factor_shrink=factor_shrink)


def _kpis(res, n_buy: int, n_sell: int, turnover: float, horizon: int) -> None:
    ist, ziel = res.wealth_ist, res.wealth_ziel
    start = ziel["start"]
    k = st.columns(4)
    k[0].metric("Depotwert heute", _eur(start), border=True)
    k[1].metric(f"Erwartet in {horizon} J. (Median)", _eur(ziel["p50"]),
                f"{_pct(ziel['p50'] / start - 1, True)} · Ist {_eur(ist['p50'])}", border=True)
    k[2].metric("Pessimistisch (P5)", _eur(ziel["p5"]), f"{_pct(ziel['p5'] / start - 1, True)} ggü. heute",
                delta_color="normal" if ziel["p5"] >= start else "inverse", border=True)
    k[3].metric("Orders", f"{n_sell} Verkauf · {n_buy} Kauf", f"Umschlag {_eur(turnover)}",
                delta_color="off", border=True)


def _order_card(row: pd.Series, buy: bool) -> None:
    with st.container(border=True):
        a, b = st.columns([3, 2])
        badge = ":blue-background[**KAUF**]" if buy else ":orange-background[**VERKAUF**]"
        a.markdown(f"{badge} &nbsp; **{_short(row['Firma'])}**")
        b.markdown(f"<div style='text-align:right;font-size:1.15rem;font-weight:600'>{_eur(abs(row['Order €']))}</div>",
                   unsafe_allow_html=True)
        st.caption(f"{abs(row['Order Stück']):.6g} Stück zu {_eur(row['Kurs €'], 2)} · Gewicht "
                   f"{_pct(row['Ist-Gewicht'])} → **{_pct(row['Ziel-Gewicht'])}** · Modell "
                   f"{_pct(row['Upside P50'], True)}, mit Mischung {_pct(row['E[R] 3J (Mix)'], True)} in 3 J.")


def _orders_tab(t: pd.DataFrame) -> None:
    o = t[(t["Firma"] != _CASH) & (t["Order €"].abs() > 0.5)]
    if o.empty:
        st.success("Keine Orders nötig – das Depot liegt innerhalb der Mindestorder am Ziel.", icon="✅")
        return
    sells, buys = o[o["Order €"] < 0].sort_values("Order €"), o[o["Order €"] > 0].sort_values("Order €", ascending=False)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"##### Verkaufen · {_eur(-sells['Order €'].sum())}")
        for _, r in sells.iterrows():
            _order_card(r, buy=False)
        if sells.empty:
            st.caption("–")
    with right:
        st.markdown(f"##### Kaufen · {_eur(buys['Order €'].sum())}")
        for _, r in buys.iterrows():
            _order_card(r, buy=True)
        if buys.empty:
            st.caption("–")
    cash_row = t[t["Firma"] == _CASH].iloc[0]
    st.caption(f"Cash: {_pct(cash_row['Ist-Gewicht'])} → {_pct(cash_row['Ziel-Gewicht'])} "
               f"({_eur(cash_row['Order €'])} durch die Orders). Orders in Bruchstücken, Käufe überziehen das Konto nie.")


def _valuations_table(t: pd.DataFrame, ages: dict) -> None:
    v = t[t["Firma"] != _CASH].copy()
    v["Alter (Tage)"] = v["Firma"].map(ages)
    v = v[["Firma", "Stand", "Alter (Tage)", "Kurs €", "Wert P50 €", "Upside P50", "EV/EBITDA", "Faktor ×", "P(Wert>Kurs)",
           "E[R] 3J (Mix)", "Vola p. a.", "Gruppe", "Ist-Gewicht", "Ziel-Gewicht"]]
    st.dataframe(
        v, hide_index=True, width="stretch",
        column_config={
            "Kurs €": st.column_config.NumberColumn(format="%.2f €"),
            "Wert P50 €": st.column_config.NumberColumn(format="%.2f €"),
            "Upside P50": st.column_config.NumberColumn("Modell-Potenzial", format="percent"),
            "EV/EBITDA": st.column_config.NumberColumn("EV/EBITDA (Markt)", format="%.1f×"),
            "Faktor ×": st.column_config.NumberColumn("Value-Korrektur", format="×%.2f",
                                                      help="Multiplikator auf den Modellwert (Regler s)."),
            "P(Wert>Kurs)": st.column_config.ProgressColumn("P(Wert > Kurs, korrigiert)", min_value=0.0, max_value=1.0, format="percent"),
            "E[R] 3J (Mix)": st.column_config.NumberColumn("Erwartet 3 J. (Mix)", format="percent"),
            "Vola p. a.": st.column_config.NumberColumn(format="percent"),
            "Ist-Gewicht": st.column_config.NumberColumn(format="percent"),
            "Ziel-Gewicht": st.column_config.ProgressColumn("Ziel", min_value=0.0, max_value=0.5, format="percent"),
        })


# ── Seite ─────────────────────────────────────────────────────────────────────────────────

def render_portfolio(container) -> None:
    with container:
        st.markdown("## 💼 Portfolio-Balancing")
        path = os.environ.get("SOTP_DEPOT", "depot.json")
        with st.popover("📁 Depot-Datei"):
            path = st.text_input("Pfad", path, key="pf_path")
        if st.session_state.get("pf_loaded") != path:
            try:
                st.session_state.pf_depot = load_depot(path)
            except (OSError, ValueError) as e:  # JSONDecodeError ist ein ValueError
                st.error(f"Depot-Datei nicht lesbar: {e}")
                return
            st.session_state.pf_loaded = path
        depot = st.session_state.pf_depot
        folder0 = st.session_state.get("pf_folder") or depot.get("bewertungen_ordner", "")
        vals = find_valuations(folder0) if folder0 and Path(folder0).is_dir() else {}
        files = {f.name: f for fs in vals.values() for f in fs}

        tabs = st.tabs(["📋 Orders", "⚖️ Allokation", "🔎 Bewertungen", "🎲 Risiko", "✏️ Depot"])
        with tabs[4]:
            positions, cash, folder = _depot_tab(depot, path, vals, files)
        if not vals:
            st.warning("Keine Bewertungen im Ordner gefunden – Ordner im Reiter „Depot“ prüfen.")
            return

        p = _settings()
        sig = hashlib.sha1(json.dumps([positions, cash, folder, p.__dict__], sort_keys=True, default=str)
                           .encode()).hexdigest()
        res_state = st.session_state.get("pf_result")
        stale = res_state is not None and res_state["sig"] != sig
        c1, c2 = st.columns([1, 3])
        run = c1.button("📐 Zielgewichte berechnen", type="primary", width="stretch", disabled=not positions)
        if stale:
            c2.warning("Eingaben geändert – Ergebnis unten ist veraltet.", icon="⚠️")
        elif res_state:
            c2.caption(f"Berechnet mit p = {_pct(p.p_market)}, ρ {str(p.rho_world).replace('.', ',')}, s {str(p.factor_shrink).replace('.', ',')}, Kelly {p.fraction:g}, max. {_pct(p.max_w)} je Titel, "
                       f"{_pct(p.group_cap)} je Gruppe, "
                       f"Cash-Zins {_pct(p.rf)}. Beträge in EUR, USD zum aktuellen Kurs.")

        if run:
            with st.status("Bewertungen neu simulieren, Kurse laden, optimieren …", expanded=False) as s:
                try:
                    valuations, ages = [], {}
                    for pos in positions:
                        f = files[pos["datei"]] if pos["datei"] else vals[pos["firma"]][-1]
                        valuations.append(_simulate(str(f), f.stat().st_mtime))
                        try:
                            ages[pos["firma"]] = (date.today() - date.fromisoformat(valuations[-1]["stichtag"])).days
                        except ValueError:
                            ages[pos["firma"]] = None
                    closes = _closes(tuple(sorted({q["ticker"] for q in positions})))
                    res = rebalance(positions, valuations, closes, cash, p)
                except (ValueError, KeyError) as e:
                    s.update(label="Fehler", state="error")
                    st.error(str(e))
                    return
                s.update(label="Fertig", state="complete")
            st.session_state.pf_result = res_state = {"sig": sig, "res": res, "ages": ages, "p": p}
            stale = False

        if not res_state:
            with tabs[0]:
                st.info("Annahmen prüfen und **Zielgewichte berechnen** klicken. Positionen pflegst du im Reiter „Depot“.",
                        icon="👆")
            return

        res, ages, pr = res_state["res"], res_state["ages"], res_state["p"]
        t = res.table
        o = t[(t["Firma"] != _CASH) & (t["Order €"].abs() > 0.5)]
        _kpis(res, int((o["Order €"] > 0).sum()), int((o["Order €"] < 0).sum()),
              float(o["Order €"].abs().sum()), pr.horizon_years)
        if res.warnings:
            with st.expander(f"ℹ️ Hinweise ({len(res.warnings)})"):
                for w in res.warnings:
                    st.markdown(f"- {w}")

        with tabs[0]:
            _orders_tab(t)
        with tabs[1]:
            st.plotly_chart(allocation_chart(t), width="stretch", config={"displayModeBar": False})
            st.caption("Ziel nach Fractional Kelly (CRRA) auf der Szenario-Mischung; Grenzen: max. Gewicht je Titel, "
                       "kein Hebel. Titel, deren Order an der Mindestorder scheitert, bleiben auf dem Ist-Gewicht.")
        with tabs[2]:
            st.plotly_chart(upside_chart(t), width="stretch", config={"displayModeBar": False})
            st.caption(f"Balken: Modell-Potenzial (Wert P50 / Kurs − 1). Raute: erwartete Rendite in {pr.horizon_years} "
                       f"Jahren nach Mischung mit p = {_pct(pr.p_market)} „Markt hat recht“ (Marktrendite "
                       f"{_pct(pr.rf + pr.erp)} p. a.).")
            _valuations_table(t, ages)
        with tabs[3]:
            st.plotly_chart(wealth_chart(res.wealth_samples, res.wealth_ist, res.wealth_ziel, pr.horizon_years),
                            width="stretch", config={"displayModeBar": False})
            ist, ziel = res.wealth_ist, res.wealth_ziel
            st.dataframe(pd.DataFrame({
                "": ["Ist-Depot", "Ziel-Depot"],
                "P5": [_eur(ist["p5"]), _eur(ziel["p5"])], "Median": [_eur(ist["p50"]), _eur(ziel["p50"])],
                "P95": [_eur(ist["p95"]), _eur(ziel["p95"])], "Mittel": [_eur(ist["mean"]), _eur(ziel["mean"])],
            }), hide_index=True, width="stretch")
            st.markdown("##### Korrelation der Wochenrenditen (EUR, 3 Jahre)")
            st.plotly_chart(corr_heatmap(res.corr), width="stretch", config={"displayModeBar": False})
