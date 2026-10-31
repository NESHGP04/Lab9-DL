"""Sensibilidad a la semilla: reentrena LightGBM (mejor modelo global) y el MLP (mejor modelo de deep learning de A por
validación) con 3 semillas (42, 43, 44), mismo protocolo que los modelos finales (datos hasta 2014-09-30, backtest de 92
orígenes), y reporta media y desviación estándar del MASE en prueba.

Uso:  python run_semillas.py   -> results/semillas.csv (una fila por modelo y semilla) y results/semillas_resumen.csv
No guarda parquets en results/preds para no contaminar la tabla comparativa (van a results/semillas_preds/).
"""
from __future__ import annotations

import gc

import pandas as pd
import torch

import common as C
import datos as D
import modelos_clasicos as M
import modelos_torch as T

SEEDS = [42, 43, 44]
OUT = C.ROOT / "results" / "semillas_preds"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    panel = C.load_panel()
    wide = D.panel_to_wide(panel)
    scaler = D.SeriesScaler().fit(wide)
    wn, ms = scaler.transform(wide), C.mase_scales(panel)
    series = wide.columns.tolist()

    summ = pd.read_csv(C.ROOT / "results" / "iteraciones_A_resumen.csv")
    best = summ[summ.modelo == "mlp"].sort_values("val_MASE").iloc[0]
    L, n_ep = int(best["L"]), int(best["mejor_epoca"])
    grid = pd.read_csv(C.ROOT / "results" / "lgbm_grid.csv").sort_values("val_MAE_norm").iloc[0]
    n_trees = int(grid["mejor_iter"])

    rows = []

    def record(name: str, modelo: str, seed: int, pred: pd.DataFrame) -> None:
        pred.assign(modelo=name).to_parquet(OUT / f"{name}.parquet", index=False)
        r = C.evaluate(OUT / f"{name}.parquet", panel)["global"]
        rows.append({"modelo": modelo, "semilla": seed, **r})
        print(name, round(r["MASE"], 4), flush=True)
        pd.DataFrame(rows).to_csv(C.ROOT / "results" / "semillas.csv", index=False)

    # 1) MLP primero (usa la GPU): la semilla afecta la inicialización, el dropout y el orden de los lotes.
    #    Va antes que LightGBM y se libera memoria entre corridas: con LightGBM en memoria la GPU del Mac se quedó sin memoria.
    for seed in SEEDS:
        net = T.build("mlp", L)
        T.fit_torch(net, wn, scaler, ms, L, split="trainval", fixed_epochs=n_ep, seed=seed, verbose=False)
        record(f"mlp_s{seed}", "mlp", seed, C.backtest(T.TorchForecaster("mlp", net, scaler, L, series), panel))
        del net
        gc.collect()
        if T.DEVICE.type == "mps":
            torch.mps.empty_cache()

    # 2) LightGBM global: la semilla afecta el bagging y feature_fraction
    for seed in SEEDS:
        lgb = M.LGBMGlobal({"objective": "l1", "num_leaves": 63, "seed": seed}, n_estimators=n_trees)
        lgb.fit(panel[panel["ds"] <= C.VAL_END])
        record(f"lightgbm_global_s{seed}", "lightgbm_global", seed, C.backtest(lgb, panel))
        del lgb
        gc.collect()

    df = pd.DataFrame(rows)
    resumen = df.groupby("modelo")[["MAE", "WAPE", "MASE"]].agg(["mean", "std"]).round(4)
    resumen.to_csv(C.ROOT / "results" / "semillas_resumen.csv")
    print(resumen.to_string())
