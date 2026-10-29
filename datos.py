from __future__ import annotations

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import common as C

RAW_PATH = C.ROOT / "LD2011_2014.txt"
HOURLY_ALL_PATH = C.DATA_DIR / "hourly_all.parquet"   # 370 clientes, cache local (gitignored)

DST_SPRING = pd.to_datetime(["2012-03-25", "2013-03-31", "2014-03-30"])
DST_AUTUMN = pd.to_datetime(["2012-10-28", "2013-10-27", "2014-10-26"])
DST_HOUR = 1  # franja 01:00-02:00


# ---------------------------------------------------------------- carga y limpieza
def load_hourly_wide(force: bool = False) -> pd.DataFrame:
    """kWh por hora, ancho (ds x cliente), 2012-01-01..2014-12-31, 370 clientes."""
    if HOURLY_ALL_PATH.exists() and not force:
        return pd.read_parquet(HOURLY_ALL_PATH)
    raw = pd.read_csv(RAW_PATH, sep=";", decimal=",", index_col=0, parse_dates=True)
    # kW cada 15 min -> kWh: suma de 4 registros / 4. NaN si falta alguno.
    hourly = raw.resample("h", closed="right", label="left").sum(min_count=4) / 4
    hourly = hourly.loc[C.PERIOD_START:C.PERIOD_END]
    C.DATA_DIR.mkdir(exist_ok=True)
    hourly.to_parquet(HOURLY_ALL_PATH)
    return hourly


def fix_dst(wide: pd.DataFrame) -> pd.DataFrame:
    """Marzo: la hora vacía se interpola entre vecinas. Octubre: la hora acumulada se divide en 2."""
    w = wide.copy()
    spring = [d + pd.Timedelta(hours=DST_HOUR) for d in DST_SPRING]
    autumn = [d + pd.Timedelta(hours=DST_HOUR) for d in DST_AUTUMN]
    w.loc[spring] = np.nan
    w = w.interpolate(limit=1, limit_area="inside")
    w.loc[autumn] = w.loc[autumn] / 2
    return w


def complete_clients(wide: pd.DataFrame) -> pd.Index:
    """Clientes con datos desde 2012-01-01 00:00 (consumo > 0 en el primer instante)
    y sin racha de ceros mayor a 7 días."""
    active_at_start = wide.columns[wide.iloc[0] > 0]
    zero_run = (wide[active_at_start] == 0).astype(int)
    longest = zero_run.apply(lambda s: s.groupby((s == 0).cumsum()).sum().max())
    return active_at_start[longest <= 24 * 7]


def pick_subset(clients: pd.Index, n: int = C.N_CLIENTES, seed: int = C.SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.array(clients), size=n, replace=False).tolist())


def build_panel(force: bool = False) -> pd.DataFrame:
    """Genera data/panel.parquet (unique_id, ds, y) y data/clientes.json."""
    if C.PANEL_PATH.exists() and not force:
        return pd.read_parquet(C.PANEL_PATH)
    wide = fix_dst(load_hourly_wide(force))
    clients = pick_subset(complete_clients(wide))
    C.CLIENTES_PATH.write_text(json.dumps(clients))
    long = (wide[clients].rename_axis("ds").reset_index()
            .melt(id_vars="ds", var_name="unique_id", value_name="y")
            .sort_values(["unique_id", "ds"]).reset_index(drop=True))
    long.to_parquet(C.PANEL_PATH, index=False)
    return long


def panel_to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="ds", columns="unique_id", values="y")


# ---------------------------------------------------------------- calendario
def pt_holidays(years=(2012, 2013, 2014)) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(holidays.Portugal(years=years).keys()))


CAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "holiday"]


def calendar_features(ds: pd.DatetimeIndex) -> pd.DataFrame:
    """Covariables conocidas a futuro: ciclos seno/coseno (hora, día de semana, mes) + feriado.
    Seno/coseno respeta que 23:00 y 00:00 son vecinas, sin añadir parámetros (a diferencia de one-hot/embeddings)."""
    hol = pt_holidays(tuple(sorted(set(ds.year))))
    f = pd.DataFrame(index=ds)
    for name, val, period in [("hour", ds.hour, 24), ("dow", ds.dayofweek, 7), ("month", ds.month - 1, 12)]:
        f[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
    f["holiday"] = ds.normalize().isin(hol).astype(float)
    return f[CAL_COLS]


# ---------------------------------------------------------------- normalización
class SeriesScaler:
    """z-score por serie con estadísticos SOLO de entrenamiento (evita la fuga de escalar con toda la serie)."""

    def fit(self, wide: pd.DataFrame, end: pd.Timestamp = C.TRAIN_END) -> "SeriesScaler":
        tr = wide.loc[:end]
        self.mean_, self.std_ = tr.mean(), tr.std().replace(0, 1.0)
        return self

    def transform(self, wide: pd.DataFrame) -> pd.DataFrame:
        return (wide - self.mean_) / self.std_

    def inverse(self, wide: pd.DataFrame) -> pd.DataFrame:
        return wide * self.std_ + self.mean_


# ---------------------------------------------------------------- Dataset de ventanas
class WindowDataset(Dataset):
    """Ventanas deslizantes sobre TODAS las series.

    X: (L, 1 + n_cal)   canal 0 = consumo normalizado, resto = calendario del pasado
    y: (H,)             consumo normalizado a pronosticar
    xf (si return_future): (H, n_cal) calendario de las H horas a pronosticar (conocido a futuro)

    split define qué ventanas entran según dónde cae el OBJETIVO (no el input):
      train: objetivo <= TRAIN_END | val: objetivo en validación | test: objetivo en prueba
    El input de val/test puede mirar hacia atrás a train (es pasado, no fuga).
    """

    def __init__(self, wide_norm: pd.DataFrame, split: str, L: int = C.L_DEFAULT, H: int = C.H,
                 stride: int = 1, return_future: bool = False, train_start: pd.Timestamp | None = None,
                 train_end: pd.Timestamp | None = None):
        self.L, self.H, self.return_future = L, H, return_future
        self.ds = wide_norm.index
        self.series = wide_norm.columns.tolist()
        self.values = torch.tensor(wide_norm.to_numpy(dtype=np.float32).T)       # (N, T)
        self.cal = torch.tensor(calendar_features(self.ds).to_numpy(dtype=np.float32))  # (T, F)

        lo, hi = {"train": (train_start or C.PERIOD_START, C.TRAIN_END),
                  "trainval": (train_start or C.PERIOD_START, train_end or C.VAL_END),   # entrenamiento final (datos hasta 2014-09-30)
                  "val": (C.VAL_START, C.VAL_END),
                  "test": (C.TEST_START, C.TEST_END)}[split]
        t_lo, t_hi = self.ds.get_loc(lo), self.ds.get_loc(hi)
        # s = índice del primer paso del objetivo; el input ocupa [s-L, s)
        first = t_lo + L if split in ("train", "trainval") else t_lo   # train: el input también debe caer en [train_start, TRAIN_END]
        starts = np.arange(first, t_hi - H + 2, stride)
        self.starts = starts
        self.n_series = len(self.series)

    def __len__(self):
        return self.n_series * len(self.starts)

    def origin_of(self, idx: int) -> pd.Timestamp:
        """Primer instante pronosticado de la ventana idx (el origen es ese instante)."""
        return self.ds[self.starts[idx % len(self.starts)]]

    def __getitem__(self, idx):
        i, j = divmod(idx, len(self.starts))
        s = int(self.starts[j])
        x = torch.cat([self.values[i, s - self.L:s, None], self.cal[s - self.L:s]], dim=1)
        y = self.values[i, s:s + self.H]
        if self.return_future:
            return x, self.cal[s:s + self.H], y
        return x, y


def count_windows(wide_norm: pd.DataFrame, L: int = C.L_DEFAULT, H: int = C.H) -> dict:
    return {sp: len(WindowDataset(wide_norm, sp, L, H)) for sp in ("train", "val", "test")}
