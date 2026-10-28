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


def backtest(model: ForecastModel, panel: pd.DataFrame, retrain: str | None = None,
             fit: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Origen móvil diario sobre prueba: 92 orígenes (00:00 de cada día), 24 h cada uno.

    - history = panel con ds < origen  (el modelo nunca ve el origen ni lo posterior).
    - Pesos: por defecto el modelo ya viene entrenado con datos <= 2014-09-30 (fit=False).
      fit=True lo entrena aquí con panel[ds <= VAL_END].
    - retrain="monthly": reentrena con ventana expansiva (ds < origen) el 1-oct, 1-nov y 1-dic.
    Devuelve unique_id, ds, y_hat (listo para save_preds).
    """
    panel = panel.sort_values(["unique_id", "ds"])
    if fit and retrain is None:
        model.fit(panel[panel["ds"] <= VAL_END])
    out = []
    for k, origin in enumerate(test_origins()):
        history = panel[panel["ds"] < origin]
        if retrain == "monthly" and origin.day == 1:
            model.fit(history)
        p = model.predict(history, origin)
        exp_ds = pd.date_range(origin, periods=H, freq="h")
        assert len(p) == p["unique_id"].nunique() * H and set(p["ds"].unique()) == set(exp_ds), \
            f"predict({origin}) debe devolver exactamente las {H} horas desde el origen"
        out.append(p[["unique_id", "ds", "y_hat"]])
        if verbose and k % 10 == 0:
            print(f"origen {k+1}/92 {origin:%Y-%m-%d}")
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ métricas
XMAS = (pd.Timestamp("2014-12-22 00:00"), pd.Timestamp("2014-12-28 23:00"))
RESUMEN_PATH = ROOT / "results" / "resumen.csv"


def mase_scales(panel: pd.DataFrame, m: int = 24, end: pd.Timestamp = TRAIN_END) -> pd.Series:
    """Escala de MASE por serie: MAE in-sample del seasonal naive de periodo m, solo con entrenamiento."""
    wide = panel[panel["ds"] <= end].pivot(index="ds", columns="unique_id", values="y")
    scale = (wide - wide.shift(m)).abs().mean()
    return scale.where(scale > 0)


def _agg(df: pd.DataFrame, scale: pd.Series) -> dict:
    """Métricas del panel completo: MAE/RMSE/WAPE en kWh, MAPE sobre y>0, MASE = media de MAE_i/escala_i."""
    e = df["y_hat"] - df["y"]
    pos = df["y"] > 0
    mae_i = e.abs().groupby(df["unique_id"]).mean()
    return {"MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())),
            "WAPE": e.abs().sum() / df["y"].abs().sum(),
            "MAPE": float((e[pos].abs() / df["y"][pos]).mean()), "MAPE_n_excluidos": int((~pos).sum()),
            "MASE": float((mae_i / scale.reindex(mae_i.index)).mean())}


def evaluate(preds_path: Path, panel: pd.DataFrame | None = None) -> dict:
    """Evalúa un parquet de predicciones contra los reales (escala original, kWh).

    Devuelve dict con: 'global' (MAE, RMSE, WAPE, MAPE, MASE), 'by_client' (MAE, RMSE, MASE),
    'by_h' (métricas por paso del horizonte h=1..24), 'by_hour' (por hora del día),
    'xmas' (métricas en 22-28 dic) y 'n' (filas evaluadas).
    El origen de cada pronóstico es el 00:00 del día de ds, así h = hora + 1.
    """
    panel = load_panel() if panel is None else panel
    p = pd.read_parquet(preds_path)
    df = p.merge(panel[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
    assert df["y"].notna().all(), "hay pronósticos sin valor real (¿fuera de la ventana de prueba?)"
    assert df["ds"].between(TEST_START, TEST_END).all(), "pronósticos fuera de 2014-10-01..12-31"
    assert len(df) == df["unique_id"].nunique() * len(test_origins()) * H, "faltan o sobran pronósticos"
    scale = mase_scales(panel)
    df["h"] = df["ds"].dt.hour + 1
    df["hour"] = df["ds"].dt.hour

    by_client = df.groupby("unique_id").apply(
        lambda g: pd.Series({"MAE": (g.y_hat - g.y).abs().mean(),
                             "RMSE": np.sqrt(((g.y_hat - g.y) ** 2).mean())}), include_groups=False)
    by_client["MASE"] = by_client["MAE"] / scale.reindex(by_client.index)
    xm = df[df["ds"].between(*XMAS)]
    return {"modelo": str(p["modelo"].iloc[0]), "n": len(df),
            "global": _agg(df, scale), "by_client": by_client,
            "by_h": pd.DataFrame({h: _agg(g, scale) for h, g in df.groupby("h")}).T,
            "by_hour": pd.DataFrame({h: _agg(g, scale) for h, g in df.groupby("hour")}).T,
            "xmas": _agg(xm, scale)}


def log_resumen(modelo: str, **info) -> None:
    """Agrega/actualiza la fila del modelo en results/resumen.csv (config, parámetros, tiempos, hardware...)."""
    RESUMEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cur = pd.read_csv(RESUMEN_PATH) if RESUMEN_PATH.exists() else pd.DataFrame(columns=["modelo"])
    cur = cur[cur["modelo"] != modelo]
    pd.concat([cur, pd.DataFrame([{"modelo": modelo, **info}])], ignore_index=True).to_csv(RESUMEN_PATH, index=False)
