"""Rückblick-Labor: Robustheit (Ausreißer, Jahrgänge). Liest labor_faelle.parquet aus labor_auswertung.py."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infrastructure.sec_xbrl import data_dir  # noqa: E402

d = pd.read_parquet(data_dir() / "labor_faelle.parquet")
d = d[d.am_rand <= 0.05].copy()
d["z"] = d.groupby("t").vorsprung.transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
u = "ueber_vorsichtig"


def gestutzt(s):
    return s.clip(*s.quantile([.05, .95])).mean()


for name, yrs in (("Lernen", range(2014, 2018)), ("Prüfen", range(2018, 2021))):
    x = d[d.t.isin(yrs)]
    top, bot = x[x.z == 9][u], x[x.z == 0][u]
    print(f"{name}: alle Ø {x[u].mean():+.1%} (Median {x[u].median():+.1%}) | oben Ø {top.mean():+.1%}, gestutzt "
          f"{gestutzt(top):+.1%}, Median {top.median():+.1%} | unten Ø {bot.mean():+.1%}, gestutzt {gestutzt(bot):+.1%}, "
          f"Median {bot.median():+.1%}")
    for t in yrs:
        y = x[x.t == t]
        print(f"   {t}: oben {y[y.z == 9][u].mean():+.0%} (gestutzt {gestutzt(y[y.z == 9][u]):+.0%}) / alle "
              f"{y[u].mean():+.0%} / unten {y[y.z == 0][u].mean():+.0%}")
