"""CP3 (Persona A): baselines StatsForecast + LightGBM global. Ejecutar: python run_baselines.py
Guarda results/preds/<modelo>.parquet, results/resumen.csv y results/lgbm_grid.csv."""
from __future__ import annotations

import platform
import subprocess
import sys
import time

import pandas as pd

import common as C
import modelos_clasicos as M

HW = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
    or platform.processor()


def run_stats(panel, name, sf_model, desc, n_jobs=1):
    m = M.StatsModel(name, sf_model, n_jobs=n_jobs)
    t = time.time()
    preds = C.backtest(m, panel)
    C.save_preds(preds, name)
    C.log_resumen(name, familia="Baseline", L_horas=m.window_weeks * 168, estrategia="local, reajuste por origen",
                  covariables="no", params_totales=0, params_entrenables=0, t_entrenamiento_s=0.0,
                  t_inferencia_s=round(m.infer_time, 1), hardware=HW, config=desc)
    print(f"{name}: {time.time() - t:.0f}s", flush=True)


def run_lgbm(panel):
    tr, va = panel[panel["ds"] <= C.TRAIN_END], panel[panel["ds"] <= C.VAL_END]
    grid = [("l2_h31", dict(objective="l2", num_leaves=31)), ("l2_h63", dict(objective="l2", num_leaves=63)),
            ("l2_h127", dict(objective="l2", num_leaves=127, min_child_samples=50)),
            ("l1_h63", dict(objective="l1", num_leaves=63))]
    rows = []
    for name, params in grid:
        m = M.LGBMGlobal(params, n_estimators=2000).fit(tr, valid=va)
        rows.append({"config": name, **params, "mejor_iter": m.best_iteration, "val_MAE_norm": m.val_mae_norm,
                     "t_s": round(m.train_time, 1)})
        print(rows[-1], flush=True)
    grid_df = pd.DataFrame(rows).sort_values("val_MAE_norm")
    grid_df.to_csv(C.ROOT / "results" / "lgbm_grid.csv", index=False)
    best = grid_df.iloc[0]
    best_params = dict(next(p for n, p in grid if n == best["config"]))
    n_iter = int(best["mejor_iter"])
    print("mejor:", best["config"], n_iter, flush=True)

    # modelo final (meses de historia = todos) y variantes con 3 / 12 / 30 meses para la figura 3 (sección 5.2)
    for tag, months in [("lightgbm_global", None), ("lightgbm_global_hist03m", 3),
                        ("lightgbm_global_hist12m", 12), ("lightgbm_global_hist30m", 30)]:
        hist = panel[panel["ds"] <= C.VAL_END]
        if months:
            hist = hist[hist["ds"] > C.VAL_END - pd.DateOffset(months=months)]
        m = M.LGBMGlobal(best_params, n_estimators=n_iter).fit(hist)
        m.name = tag
        t = time.time()
        C.save_preds(C.backtest(m, panel), tag)
        C.log_resumen(tag, familia="Árboles", L_horas="1-24 y 168 (rezagos)", estrategia="directa global (h como feature)",
                      covariables="sí (calendario)", params_totales=m.n_params, params_entrenables=m.n_params,
                      t_entrenamiento_s=round(m.train_time, 1), t_inferencia_s=round(m.infer_time, 1), hardware=HW,
                      meses_historia=months or 33, config=f"{best['config']}, {n_iter} árboles")
        print(tag, f"entreno {m.train_time:.0f}s, backtest {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    from statsforecast.models import MSTL, Naive, SeasonalNaive
    panel = C.load_panel()
    what = sys.argv[1:] or ["stats", "lgbm"]
    if "stats" in what:
        run_stats(panel, "naive", Naive(), "último valor")
        run_stats(panel, "snaive_24h", SeasonalNaive(24), "mismo horario del día anterior")
        run_stats(panel, "snaive_168h", SeasonalNaive(168), "mismo horario de la semana anterior")
        run_stats(panel, "mstl", MSTL(season_length=[24, 168]), "MSTL 24+168, tendencia AutoETS", n_jobs=-1)
    if "lgbm" in what:
        run_lgbm(panel)
