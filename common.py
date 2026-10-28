"""Contrato compartido del Lab 9. Lo edita Persona A; B lo importa.

Panel: DataFrame largo con columnas unique_id, ds, y (kWh por hora).
Predicciones: results/preds/<modelo>.parquet con unique_id, ds, modelo, y_hat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "results" / "preds"

SEED = 42
N_CLIENTES = 100          # mismo subconjunto para todos los modelos
H = 24                    # horizonte day-ahead
L_DEFAULT = 168

PERIOD_START = pd.Timestamp("2012-01-01 00:00")
TRAIN_END = pd.Timestamp("2014-06-30 23:00")
VAL_START, VAL_END = pd.Timestamp("2014-07-01 00:00"), pd.Timestamp("2014-09-30 23:00")
TEST_START, TEST_END = pd.Timestamp("2014-10-01 00:00"), pd.Timestamp("2014-12-31 23:00")
PERIOD_END = TEST_END

PANEL_PATH = DATA_DIR / "panel.parquet"
CLIENTES_PATH = DATA_DIR / "clientes.json"
PRED_COLS = ["unique_id", "ds", "modelo", "y_hat"]


def test_origins() -> pd.DatetimeIndex:
    """92 orígenes: 00:00 de cada día de prueba. Cada origen usa datos hasta 23:00 del día anterior."""
    return pd.date_range(TEST_START.normalize(), TEST_END.normalize(), freq="D")


def load_panel() -> pd.DataFrame:
    """[A] Panel horario limpio (unique_id, ds, y). Lo genera 01_datos.ipynb."""
    return pd.read_parquet(PANEL_PATH)


def split_panel(panel: pd.DataFrame):
    """Devuelve (train, val, test) por fecha."""
    ds = panel["ds"]
    return (panel[ds <= TRAIN_END],
            panel[(ds >= VAL_START) & (ds <= VAL_END)],
            panel[ds >= TEST_START])


def make_synthetic_panel(n_clients: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Panel falso con el mismo formato, para que B programe modelos antes de tener datos reales."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range(PERIOD_START, PERIOD_END, freq="h")
    hour, dow = ds.hour.values, ds.dayofweek.values
    frames = []
    for i in range(n_clients):
        base = rng.uniform(5, 500)
        y = base * (1 + 0.4 * np.sin(2 * np.pi * (hour - 6) / 24) - 0.15 * (dow >= 5)) \
            + rng.normal(0, 0.05 * base, len(ds))
        frames.append(pd.DataFrame({"unique_id": f"MT_{i:03d}", "ds": ds, "y": np.clip(y, 0, None)}))
    return pd.concat(frames, ignore_index=True)


class ForecastModel:
    """Interfaz que cumplen todos los modelos propios (backtest la usa)."""
    name = "modelo"

    def fit(self, train: pd.DataFrame) -> "ForecastModel":
        raise NotImplementedError

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        """history: panel con ds < origin (solo pasado). Devuelve unique_id, ds, y_hat (H filas por cliente)."""
        raise NotImplementedError


def save_preds(df: pd.DataFrame, modelo: str) -> Path:
    """Guarda predicciones validando el esquema acordado."""
    df = df.assign(modelo=modelo)[PRED_COLS]
    n_cli, n_orig = df["unique_id"].nunique(), len(test_origins())
    assert len(df) == n_cli * n_orig * H, f"se esperaban {n_cli*n_orig*H} filas, hay {len(df)}"
    assert df["y_hat"].notna().all(), "y_hat con NaN"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{modelo}.parquet"
    df.to_parquet(path, index=False)
    return path


def backtest(model: ForecastModel, panel: pd.DataFrame) -> pd.DataFrame:
    """[A] Origen móvil diario sobre prueba (92 orígenes). TODO en CP2."""
    raise NotImplementedError


def evaluate(preds_path: Path, panel: pd.DataFrame | None = None) -> dict:
    """[A] MAE, RMSE, WAPE, MAPE, MASE + error por h, por hora y semana 22-28 dic. TODO en CP2."""
    raise NotImplementedError
