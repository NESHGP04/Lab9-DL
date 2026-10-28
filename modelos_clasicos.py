"""Baselines estadísticos (StatsForecast) y LightGBM global. Persona A, CP3.

Ambos cumplen la interfaz common.ForecastModel (fit / predict(history, origin)).
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import common as C
import datos as D


# ------------------------------------------------------------------ StatsForecast
class StatsModel(C.ForecastModel):
    """Modelo local de StatsForecast. Se reajusta en cada origen con las últimas `window_weeks`
    semanas de historia (no hay pesos globales que entrenar una sola vez)."""

    def __init__(self, name: str, sf_model, window_weeks: int = 8, n_jobs: int = 1):
        self.name, self.sf_model, self.window_weeks, self.n_jobs = name, sf_model, window_weeks, n_jobs
        self.infer_time = 0.0

    def fit(self, train: pd.DataFrame) -> "StatsModel":
        return self

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        from statsforecast import StatsForecast
        t0 = time.time()
        tail = history[history["ds"] >= origin - pd.Timedelta(weeks=self.window_weeks)]
        f = StatsForecast(models=[self.sf_model], freq="h", n_jobs=self.n_jobs).forecast(df=tail, h=C.H)
        self.infer_time += time.time() - t0
        out = f.rename(columns={f.columns[-1]: "y_hat"})[["unique_id", "ds", "y_hat"]]
        out["y_hat"] = out["y_hat"].clip(lower=0)
        return out


# ------------------------------------------------------------------ LightGBM global
LAGS_ORIGIN = np.arange(1, 25)   # y[origen-1h], ..., y[origen-24h]  (últimas 24 h antes del pronóstico)


def _features(Wn: np.ndarray, cal: np.ndarray, origins: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Features directas por (origen, cliente, paso h).

    Wn: (T, N) consumo normalizado; cal: (T, F) calendario; origins: índices (en T) de los orígenes 00:00.
    Todo se construye con datos < origen:
      - rezagos 1..24 relativos al ORIGEN (las 24 h previas), iguales para los 24 pasos;
      - rezago 168 y 24 del instante objetivo t = origen + j  (t-168 y t-24 caen siempre antes del origen);
      - nivel: media de las últimas 168 h; calendario del instante objetivo; paso j.
    """
    O, N, J = len(origins), Wn.shape[1], np.arange(C.H)
    lag_o = Wn[origins[:, None] - LAGS_ORIGIN[None, :]]                    # (O, 24, N)
    lag_o = np.broadcast_to(lag_o.transpose(0, 2, 1)[:, :, None, :], (O, N, C.H, 24))
    t = origins[:, None] + J[None, :]                                       # (O, 24)
    lag168 = Wn[t - 168].transpose(0, 2, 1)[..., None]                      # (O, N, 24, 1)
    lag24 = Wn[t - 24].transpose(0, 2, 1)[..., None]
    csum = np.vstack([np.zeros((1, N), dtype=np.float64), np.cumsum(Wn, axis=0, dtype=np.float64)])
    lvl = ((csum[origins] - csum[origins - 168]) / 168).astype(np.float32)  # (O, N)
    lvl = np.broadcast_to(lvl[:, :, None, None], (O, N, C.H, 1))
    c = np.broadcast_to(cal[t][:, None, :, :], (O, N, C.H, cal.shape[1]))
    h = np.broadcast_to(J[None, None, :, None].astype(np.float32), (O, N, C.H, 1))
    X = np.concatenate([lag_o, lag168, lag24, lvl, c, h], axis=-1).reshape(-1, 24 + 3 + cal.shape[1] + 1)
    names = [f"lag_{k}" for k in LAGS_ORIGIN] + ["lag168_t", "lag24_t", "nivel168"] + D.CAL_COLS + ["h"]
    return X.astype(np.float32), names


def _target(Wn: np.ndarray, origins: np.ndarray) -> np.ndarray:
    t = origins[:, None] + np.arange(C.H)[None, :]
    return Wn[t].transpose(0, 2, 1).reshape(-1).astype(np.float32)           # (O, N, 24) en el mismo orden


class LGBMGlobal(C.ForecastModel):
    """LightGBM global (un modelo para todos los clientes, estrategia directa con 'h' como feature)."""
    name = "lightgbm_global"

    def __init__(self, params: dict | None = None, n_estimators: int = 500):
        self.params = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
                       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                       "verbosity": -1, "seed": C.SEED, "num_threads": 10, **(params or {})}
        self.n_estimators = n_estimators
        self.train_time = 0.0
        self.infer_time = 0.0

    # --- preparación
    def _prep(self, panel: pd.DataFrame):
        wide = D.panel_to_wide(panel)
        self.series = wide.columns.tolist()
        Wn = self.scaler.transform(wide).to_numpy(dtype=np.float32)
        cal = D.calendar_features(wide.index).to_numpy(dtype=np.float32)
        return wide, Wn, cal

    def _origins(self, ds: pd.DatetimeIndex, lo: pd.Timestamp, hi: pd.Timestamp) -> np.ndarray:
        """Índices de los orígenes 00:00 con los 24 pasos objetivo dentro de [lo, hi] e historia de 168 h."""
        days = pd.date_range(lo.normalize(), hi.normalize() - pd.Timedelta(days=0), freq="D")
        idx = np.array([ds.get_loc(d) for d in days if d + pd.Timedelta(hours=C.H - 1) <= hi])
        return idx[idx >= 168]

    def dataset(self, panel: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp):
        wide, Wn, cal = self._prep(panel)
        o = self._origins(wide.index, lo, hi)
        X, names = _features(Wn, cal, o)
        return X, _target(Wn, o), names

    # --- entrenamiento
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None, early_stopping: int = 50) -> "LGBMGlobal":
        """train: panel con ds <= fin del bloque de entrenamiento. Si se da `valid` (panel completo que incluye
        validación como pasado), usa early stopping sobre ese bloque; si no, entrena n_estimators árboles."""
        t0 = time.time()
        wide_tr = D.panel_to_wide(train)
        self.scaler = D.SeriesScaler().fit(wide_tr, end=wide_tr.index[-1] if valid is None else C.TRAIN_END)
        end_tr = train["ds"].max()
        X, y, self.feature_names = self.dataset(train, wide_tr.index[0], end_tr)
        dtr = lgb.Dataset(X, y, feature_name=self.feature_names, free_raw_data=False)
        callbacks, valid_sets = [], None
        if valid is not None:
            Xv, yv, _ = self.dataset(valid, C.VAL_START, C.VAL_END)
            valid_sets = [lgb.Dataset(Xv, yv, reference=dtr)]
            callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
        self.model = lgb.train({**self.params, "metric": "l1"}, dtr, num_boost_round=self.n_estimators,
                               valid_sets=valid_sets, callbacks=callbacks)
        self.best_iteration = self.model.best_iteration if valid is not None else self.n_estimators
        self.val_mae_norm = self.model.best_score["valid_0"]["l1"] if valid is not None else None
        self.train_time = time.time() - t0
        return self

    @property
    def n_params(self) -> int:
        """Hojas totales del ensamble (aproximación a 'parámetros' de un modelo de árboles)."""
        df = self.model.trees_to_dataframe()
        return int(df["split_feature"].isna().sum())

    # --- pronóstico
    def predict(self, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        t0 = time.time()
        tail = history[history["ds"] >= origin - pd.Timedelta(hours=170)]
        wide = D.panel_to_wide(tail).reindex(columns=self.series)
        # añade las H horas del futuro como NaN solo para el calendario (el consumo futuro nunca se lee)
        full_ds = pd.date_range(wide.index[0], origin + pd.Timedelta(hours=C.H - 1), freq="h")
        Wn = self.scaler.transform(wide.reindex(full_ds)).to_numpy(dtype=np.float32)
        cal = D.calendar_features(full_ds).to_numpy(dtype=np.float32)
        o = np.array([full_ds.get_loc(origin)])
        X, _ = _features(np.nan_to_num(Wn, nan=0.0), cal, o)
        yhat = self.model.predict(X, num_iteration=self.best_iteration).reshape(len(self.series), C.H)
        yhat = yhat * self.scaler.std_.reindex(self.series).to_numpy()[:, None] \
            + self.scaler.mean_.reindex(self.series).to_numpy()[:, None]
        self.infer_time += time.time() - t0
        ds = pd.date_range(origin, periods=C.H, freq="h")
        return pd.DataFrame({"unique_id": np.repeat(self.series, C.H), "ds": np.tile(ds, len(self.series)),
                             "y_hat": np.clip(yhat.reshape(-1), 0, None)})
