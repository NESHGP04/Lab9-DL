"""CP5 (Persona A): reentrenamiento mensual con ventana expansiva (1-oct, 1-nov, 1-dic) para el mejor modelo de deep
learning de A (MLP, elegido por MASE de validación) y para LightGBM. Ejecutar: python run_reentreno.py
Guarda results/preds/<modelo>_reentreno.parquet. El protocolo es common.backtest(model, panel, retrain="monthly")."""
from __future__ import annotations

import pandas as pd

import common as C
import datos as D
import modelos_clasicos as M
import modelos_torch as T
from run_baselines import HW

if __name__ == "__main__":
    panel = C.load_panel()
    wide = D.panel_to_wide(panel)
    scaler = D.SeriesScaler().fit(wide)
    wn, ms = scaler.transform(wide), C.mase_scales(panel)
    series = wide.columns.tolist()

    # mejor DL de A por validación -> MLP; épocas y L tomados de la tabla de iteraciones
    summ = pd.read_csv(C.ROOT / "results" / "iteraciones_A_resumen.csv")
    best = summ[summ.modelo == "mlp"].sort_values("val_MASE").iloc[0]
    L, n_ep = int(best["L"]), int(best["mejor_epoca"])
    mlp = T.RetrainableTorch("mlp_reentreno", "mlp", L, n_ep, wn, scaler, ms, series)
    C.save_preds(C.backtest(mlp, panel, retrain="monthly"), "mlp_reentreno")
    print("mlp_reentreno: fits hasta", [str(e) for e in mlp.fit_ends], [round(t) for t in mlp.fit_times], flush=True)
    C.log_resumen("mlp_reentreno", familia="MLP", L_horas=L, estrategia="multi-salida, reentreno mensual",
                  covariables="sí", t_entrenamiento_s=round(sum(mlp.fit_times), 1), t_inferencia_s=round(mlp.inner.infer_time, 1),
                  hardware=HW, config=f"L={L}, {n_ep} épocas por reentreno, desde cero")

    # LightGBM: mismos hiperparámetros y nº de árboles que el modelo fijo
    grid = pd.read_csv(C.ROOT / "results" / "lgbm_grid.csv").sort_values("val_MAE_norm").iloc[0]
    params = {"objective": grid["objective"], "num_leaves": int(grid["num_leaves"])}
    if pd.notna(grid.get("min_child_samples")):
        params["min_child_samples"] = int(grid["min_child_samples"])
    lgb = M.LGBMGlobal(params, n_estimators=int(grid["mejor_iter"]))
    lgb.name = "lightgbm_global_reentreno"
    C.save_preds(C.backtest(lgb, panel, retrain="monthly"), lgb.name)
    C.log_resumen(lgb.name, familia="Árboles", estrategia="directa global, reentreno mensual", covariables="sí (calendario)",
                  t_entrenamiento_s=round(lgb.train_time, 1), t_inferencia_s=round(lgb.infer_time, 1), hardware=HW,
                  config=f"{grid['config']}, {int(grid['mejor_iter'])} árboles por reentreno")
    print("lightgbm_global_reentreno listo", flush=True)
