"""Modelos de librería (NeuralForecast, Nixtla): N-HiTS, TCN, PatchTST, iTransformer y DeepAR (probabilístico).

Mismo protocolo que el resto:
  - entrenamiento con datos hasta 2014-06-30 y early stopping sobre VALIDACIÓN (2014-07-01..09-30); NeuralForecast
    separa esos últimos 92 días de lo que le damos hasta 2014-09-30 con `val_size` (el modelo NO se reentrena con la validación);
  - pesos entrenados una sola vez y 92 orígenes diarios a las 00:00 con H = 24 sobre la prueba;
  - parquet en results/preds/<modelo>.parquet y fila en results/resumen.csv.
Normalización: scaler_type="standard" de la librería (estadísticos de cada ventana de entrada, no de la prueba).
Covariables de calendario (futr_exog): N-HiTS, TCN y DeepAR; PatchTST e iTransformer no las admiten.

Uso:
  python run_neuralforecast.py --probe                 # prueba rápida: 10 clientes, pocos pasos, no escribe resultados
  python run_neuralforecast.py nhits tcn patchtst      # modelos a entrenar (por defecto: todos)
  python run_neuralforecast.py deepar --max-steps=2000
  python run_neuralforecast.py --cpu patchtst          # en Mac: PatchTST/iTransformer fallan en la GPU MPS
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

import common as C
import datos as D

L = 168
VAL_DAYS = 92
MAX_STEPS = 3000
SERIES_BATCH = 8      # series por lote (en NeuralForecast batch_size = nº de SERIES, no de ventanas); 100 provoca OOM en una T4
BATCH = 256           # ventanas muestreadas por paso (windows_batch_size)
CAL = D.CAL_COLS
NAMES = ["nhits", "tcn", "patchtst", "itransformer", "deepar"]


def hardware() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "Apple GPU (MPS)" if torch.backends.mps.is_available() else "CPU"


def make_model(name: str, n_series: int, max_steps: int, val_steps: int):
    from neuralforecast.losses.pytorch import MAE, MQLoss, DistributionLoss
    from neuralforecast.models import NHITS, TCN, PatchTST, iTransformer, DeepAR
    common = dict(h=C.H, input_size=L, max_steps=max_steps, val_check_steps=val_steps, early_stop_patience_steps=5,
                  scaler_type="standard", random_seed=C.SEED, batch_size=SERIES_BATCH, learning_rate=1e-3)
    if "--cpu" in sys.argv:       # PatchTST/iTransformer fallan en MPS (Mac); en CUDA (Colab) no hace falta
        common["accelerator"] = "cpu"
    if name == "nhits":
        return NHITS(**common, valid_batch_size=SERIES_BATCH, inference_windows_batch_size=BATCH, loss=MAE(),
                     futr_exog_list=CAL, windows_batch_size=BATCH)
    if name == "tcn":
        return TCN(**common, valid_batch_size=SERIES_BATCH, loss=MAE(), futr_exog_list=CAL, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32, 64],
                   encoder_hidden_size=64, decoder_hidden_size=64)
    if name == "patchtst":
        return PatchTST(**common, valid_batch_size=SERIES_BATCH, inference_windows_batch_size=BATCH, loss=MAE(), encoder_layers=2, n_heads=4, hidden_size=64, linear_hidden_size=128,
                        patch_len=16, stride=8, windows_batch_size=BATCH)
    if name == "itransformer":
        common["batch_size"] = 32      # modelo multivariado: aquí batch_size cuenta ventanas (cada una trae las 100 series)
        return iTransformer(**common, loss=MAE(), n_series=n_series, hidden_size=128, n_heads=4, e_layers=2, d_ff=256)
    if name == "deepar":
        # P10 / P50 / P90 con una distribución Student-t; level=[80] -> bandas lo-80 / hi-80 (P10 y P90)
        return DeepAR(**common, valid_batch_size=SERIES_BATCH, inference_windows_batch_size=BATCH,
                      windows_batch_size=BATCH, loss=DistributionLoss("StudentT", level=[80]), futr_exog_list=CAL,
                      lstm_n_layers=2, lstm_hidden_size=64, trajectory_samples=100)
    raise ValueError(name)


def to_nf_frame(panel: pd.DataFrame, clients: list[str]) -> pd.DataFrame:
    cal = D.calendar_features(pd.DatetimeIndex(sorted(panel["ds"].unique()))).reset_index(names="ds")
    df = panel[panel["unique_id"].isin(clients)].merge(cal, on="ds", how="left")
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def to_preds(fc: pd.DataFrame, name: str, model_col: str) -> pd.DataFrame:
    """Convierte la salida de predict al esquema acordado."""
    out = fc[["unique_id", "ds"]].copy()
    if name == "deepar":
        out["q10"], out["q90"] = fc[f"{model_col}-lo-80"].clip(lower=0), fc[f"{model_col}-hi-80"].clip(lower=0)
        med = f"{model_col}-median" if f"{model_col}-median" in fc else model_col
        out["q50"] = fc[med].clip(lower=0)
        out["y_hat"] = out["q50"]
    else:
        out["y_hat"] = fc[model_col].clip(lower=0)
    return out


def run(name: str, df: pd.DataFrame, clients: list[str], max_steps: int, val_steps: int, probe: bool) -> None:
    from neuralforecast import NeuralForecast
    model = make_model(name, len(clients), max_steps, val_steps)
    n_par = sum(p.numel() for p in model.parameters())
    nf = NeuralForecast(models=[model], freq="h")
    # 1) entrenamiento ÚNICO con datos hasta 2014-09-30: los últimos 92 días (jul-sep) son validación (early stopping)
    t0 = time.time()
    nf.fit(df=df[df["ds"] <= C.VAL_END], val_size=VAL_DAYS * C.H)
    t_fit = time.time() - t0
    # 2) 92 orígenes diarios: solo las L horas anteriores al origen como entrada (nunca el origen ni lo posterior).
    #    Se usa predict por origen (y no cross_validation) porque TCN y DeepAR no admiten step_size > 1.
    t0, out, model_col = time.time(), [], None
    for k, origin in enumerate(C.test_origins()):
        hist = df[(df["ds"] >= origin - pd.Timedelta(hours=L)) & (df["ds"] < origin)]
        fut = df[(df["ds"] >= origin) & (df["ds"] < origin + pd.Timedelta(hours=C.H))][["unique_id", "ds"] + CAL]
        fc = nf.predict(df=hist, futr_df=fut if model.futr_exog_list else None)
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        model_col = model_col or (type(model).__name__ if type(model).__name__ in fc.columns else next(
            c for c in fc.columns if c not in ("unique_id", "ds") and "-" not in c))
        out.append(to_preds(fc, name, model_col))
        if k % 10 == 0:
            print(f"  origen {k + 1}/92 {origin:%Y-%m-%d}", flush=True)
    t_inf = time.time() - t0
    t_total = t_fit + t_inf
    pred = pd.concat(out, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    exp = pred.groupby("unique_id").size()
    assert (exp == len(C.test_origins()) * C.H).all(), "cada cliente debe tener 92 × 24 pronósticos"
    assert pred["ds"].min() == C.TEST_START and pred["ds"].max() == C.TEST_END
    if probe:
        sub = df[df["unique_id"].isin(clients)]
        m = pred.merge(sub[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
        print(f"[probe] {name}: params={n_par:,} fit={t_fit:.0f}s inf={t_inf:.0f}s MAE={np.abs(m.y_hat - m.y).mean():.2f}", flush=True)
        return
    C.save_preds(pred, name)
    C.log_resumen(name, familia={"nhits": "Transformer/MLP jerárquico", "tcn": "Convolucional", "patchtst": "Transformer",
                                 "itransformer": "Transformer", "deepar": "Probabilístico"}[name],
                  L_horas=L, estrategia="multi-salida" if name != "deepar" else "autorregresivo probabilístico (muestreo)",
                  covariables="sí (calendario)" if name in ("nhits", "tcn", "deepar") else "no",
                  params_totales=n_par, params_entrenables=n_par, t_entrenamiento_s=round(t_fit, 1),
                  t_inferencia_s=round(t_inf, 1), hardware=hardware(),
                  config=f"NeuralForecast, L={L}, max_steps={max_steps}, early stopping en validación, batch={BATCH}, "
                         f"Adam lr=1e-3, scaler standard")
    print(name, f"listo: fit {t_fit:.0f}s + inferencia {t_inf:.0f}s | params={n_par:,}", flush=True)


if __name__ == "__main__":
    probe = "--probe" in sys.argv
    max_steps = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--max-steps=")), 200 if probe else MAX_STEPS)
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or NAMES
    panel = C.load_panel()
    clients = sorted(panel["unique_id"].unique())
    if probe:
        clients = clients[:10]
    df = to_nf_frame(panel, clients)
    print(f"dispositivo: {hardware()} | {len(clients)} clientes | max_steps={max_steps}", flush=True)
    for name in names:
        if name == "itransformer" and len(clients) != C.N_CLIENTES and not probe:
            continue
        run(name, df, clients, max_steps, val_steps=max(25, max_steps // 30), probe=probe)
