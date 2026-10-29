from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

SUFIJOS_EXCLUIDOS = ("_hist03m", "_hist12m", "_hist30m", "_reentreno")
HISTORIA_MODELOS = ["snaive_168h", "lightgbm_global", "dlinear", "lstm", "patchtst"]   # figura 3 (los que existan)
CHRONOS = "chronos_bolt"


def main_models(resumen: pd.DataFrame) -> list[str]:
    return [m for m in resumen["modelo"] if not m.endswith(SUFIJOS_EXCLUIDOS) and (C.PRED_DIR / f"{m}.parquet").exists()]


def prob_metrics(path) -> dict | None:
    """Pinball loss (media sobre cuantiles disponibles, kWh) y cobertura P10-P90, si el parquet trae q10 y q90."""
    p = pd.read_parquet(path)
    qs = {int(c[1:]): c for c in p.columns if c.startswith("q") and c[1:].isdigit()}
    if not {10, 90} <= set(qs):
        return None
    y = p.merge(C.load_panel(), on=["unique_id", "ds"])
    pin = np.mean([np.mean(np.maximum(q / 100 * (y["y"] - y[c]), (q / 100 - 1) * (y["y"] - y[c]))) for q, c in qs.items()])
    return {"pinball": float(pin), "cobertura_P10_P90": float(((y["y"] >= y[qs[10]]) & (y["y"] <= y[qs[90]])).mean())}


def tabla(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = C.load_panel() if panel is None else panel
    resumen = pd.read_csv(C.RESUMEN_PATH).set_index("modelo")
    rows = []
    for m in main_models(resumen.reset_index()):
        r = C.evaluate(C.PRED_DIR / f"{m}.parquet", panel)
        info = resumen.loc[m]
        pr = prob_metrics(C.PRED_DIR / f"{m}.parquet") or {}
        # los modelos locales de StatsForecast ajustan dentro de predict(): su ajuste cuenta como entrenamiento
        local = str(info.get("estrategia", "")).startswith("local")
        rows.append({"modelo": m, "familia": info["familia"], "L (h)": info["L_horas"], "multi-paso": info["estrategia"],
                     "covariables": info["covariables"], "params totales": info["params_totales"],
                     "params entrenables": info["params_entrenables"],
                     "t_entrenamiento (s)": info["t_entrenamiento_s"] + (info["t_inferencia_s"] if local else 0.0),
                     "t_inferencia (s)": info["t_inferencia_s"], "hardware": info["hardware"],
                     "MAE": r["global"]["MAE"], "RMSE": r["global"]["RMSE"], "WAPE": r["global"]["WAPE"],
                     "MASE": r["global"]["MASE"], "MASE 22-28 dic": r["xmas"]["MASE"],
                     "pinball": pr.get("pinball"), "cobertura P10-P90": pr.get("cobertura_P10_P90")})
    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Índices no dominados (menor x y menor y), ordenados por x."""
    idx = np.argsort(x)
    front, best = [], np.inf
    for i in idx:
        if y[i] < best:
            front.append(i); best = y[i]
    return np.array(front)
