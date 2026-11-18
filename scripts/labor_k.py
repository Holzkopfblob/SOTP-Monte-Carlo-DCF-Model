"""Baustein 3: Wie viel vom vorhergesagten Vorsprung kommt an? Median- und Mittelwert-Regression der tatsächlichen
Überrendite (gegen S&P 500 TR) auf den Vorsprung (Rendite zum Kurs − ETF-Rendite damals), 5 J., Variante vorsichtig.
Konfidenzband per Bootstrap über Firmen (Firma = Cluster). Liest labor_faelle.parquet (labor_auswertung.py).

Aufruf:  python scripts/labor_k.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application.labor import lad  # noqa: E402
from infrastructure.sec_xbrl import data_dir  # noqa: E402


def ols(x, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]


def schaetze(d, f, n_boot=300, seed=0):
    x = d.vorsprung.clip(*d.vorsprung.quantile([.01, .99])).to_numpy()
    y = d.ueber_vorsichtig.to_numpy()
    b = f(x, y)
    rng = np.random.default_rng(seed)
    firmen = d.cik.unique()
    idx = {c: np.flatnonzero(d.cik.to_numpy() == c) for c in firmen}
    boot = []
    for _ in range(n_boot):
        i = np.concatenate([idx[c] for c in rng.choice(firmen, len(firmen))])
        boot.append(f(x[i], y[i]))
    lo, hi = np.percentile(np.array(boot), [5, 95], axis=0)
    return b, lo, hi


d = pd.read_parquet(data_dir() / "labor_faelle.parquet")
d = d[(d.am_rand <= 0.05) & d.ueber_vorsichtig.notna() & d.vorsprung.notna()]
for name, yrs in (("Lernen 2014–17", range(2014, 2018)), ("Prüfen 2018–20", range(2018, 2021)), ("alle", range(2014, 2021))):
    x = d[d.t.isin(yrs)]
    for art, f in (("Median", lad), ("Mittel", ols)):
        b, lo, hi = schaetze(x, f)
        print(f"{name:15} {art:6}: Achse {b[0]:+.3f} [{lo[0]:+.3f}; {hi[0]:+.3f}], k {b[1]:.2f} [{lo[1]:.2f}; {hi[1]:.2f}] "
              f"(90 %, n {len(x)})")
