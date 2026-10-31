"""Chronos-Bolt zero-shot (Amazon): modelo preentrenado que NO se entrena ni se ajusta con nuestros datos.

Para cada uno de los 92 orígenes (00:00) toma como contexto las últimas `CONTEXT` horas de cada cliente (solo datos
anteriores al origen), pronostica 24 h y devuelve los cuantiles 0.1 / 0.5 / 0.9. La mediana (q50) es el pronóstico puntual
(y_hat) y q10 / q90 permiten pinball loss y cobertura P10–P90 en la sección de resultados probabilísticos.

Uso:
  python run_chronos.py                          # chronos-bolt-small (por defecto)
  python run_chronos.py amazon/chronos-bolt-base # otro tamaño
  python run_chronos.py --probe                  # 3 orígenes y 10 clientes, no escribe resultados
Requiere:  pip install chronos-forecasting   (descarga los pesos de Hugging Face la primera vez)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

import common as C
import datos as D

CONTEXT = 1344      # 8 semanas de historia (igual que la ventana de los modelos estadísticos)
QUANTILES = [0.1, 0.5, 0.9]


def hardware() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "Apple GPU (MPS)" if torch.backends.mps.is_available() else "CPU"


def forecast(pipe, wide: pd.DataFrame, origins, batch: int = 100) -> pd.DataFrame:
    """wide: consumo horario (ds × cliente) completo. Para cada origen usa SOLO wide.loc[:origen - 1h]."""
    cols, out = wide.columns.tolist(), []
    for k, origin in enumerate(origins):
        ctx = wide.loc[origin - pd.Timedelta(hours=CONTEXT):origin - pd.Timedelta(hours=1)]
        assert ctx.index[-1] == origin - pd.Timedelta(hours=1) and len(ctx) == CONTEXT
        series = [torch.tensor(ctx[c].to_numpy(dtype=np.float32)) for c in cols]
        qs = []
        for i in range(0, len(series), batch):
            q, _ = pipe.predict_quantiles(series[i:i + batch], prediction_length=C.H, quantile_levels=QUANTILES)
            qs.append(q.cpu().numpy())                                                 # (n, 24, 3)
        q = np.concatenate(qs)
        ds = pd.date_range(origin, periods=C.H, freq="h")
        out.append(pd.DataFrame({"unique_id": np.repeat(cols, C.H), "ds": np.tile(ds, len(cols)),
                                 "q10": q[..., 0].reshape(-1), "q50": q[..., 1].reshape(-1), "q90": q[..., 2].reshape(-1)}))
        if k % 10 == 0:
            print(f"origen {k + 1}/{len(origins)} {origin:%Y-%m-%d}", flush=True)
    df = pd.concat(out, ignore_index=True)
    for c in ("q10", "q50", "q90"):
        df[c] = df[c].clip(lower=0)
    df["y_hat"] = df["q50"]
    return df


if __name__ == "__main__":
    probe = "--probe" in sys.argv
    ckpt = next((a for a in sys.argv[1:] if not a.startswith("--")), "amazon/chronos-bolt-small")
    name = "chronos_" + ckpt.split("-")[-1]
    from chronos import BaseChronosPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"        # MPS no es estable en Chronos-Bolt; CPU es suficiente
    pipe = BaseChronosPipeline.from_pretrained(ckpt, device_map=device, torch_dtype=torch.float32)
    n_par = sum(p.numel() for p in pipe.model.parameters())

    panel = C.load_panel()
    wide = D.panel_to_wide(panel)
    origins = C.test_origins()
    if probe:
        wide, origins = wide.iloc[:, :10], origins[:3]
    t0 = time.time()
    pred = forecast(pipe, wide, origins)
    t_inf = time.time() - t0
    if probe:
        m = pred.merge(panel, on=["unique_id", "ds"])
        print(f"[probe] {name}: MAE={np.abs(m.y_hat - m.y).mean():.2f} cobertura P10-P90="
              f"{((m.y >= m.q10) & (m.y <= m.q90)).mean():.2f} t={t_inf:.1f}s", flush=True)
        sys.exit()
    C.save_preds(pred, name)
    C.log_resumen(name, familia="Modelo fundacional (zero-shot)", L_horas=CONTEXT, estrategia="zero-shot, cuantiles directos",
                  covariables="no", params_totales=n_par, params_entrenables=0, t_entrenamiento_s=0.0,
                  t_inferencia_s=round(t_inf, 1), hardware=hardware(),
                  config=f"{ckpt}, contexto {CONTEXT} h, sin ajuste con los datos; q10/q50/q90")
    print(name, f"listo en {t_inf:.0f}s | params={n_par:,}", flush=True)
