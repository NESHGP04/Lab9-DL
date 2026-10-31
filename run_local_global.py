"""5.3 Local vs global: ¿un modelo por cliente o uno para todos? (10 clientes).

Local  = un modelo independiente por cada uno de 10 clientes (LightGBM, DLinear y MLP), con el mismo protocolo que el global:
         iteración con early stopping en VALIDACIÓN, reentrenamiento con datos hasta 2014-09-30 y backtest de 92 orígenes.
Global = los pronósticos ya existentes de lightgbm_global / dlinear / mlp (entrenados con los 100 clientes), restringidos a
         los mismos 10 clientes. Así la comparación es sobre exactamente las mismas series y los mismos 92 × 24 pronósticos.

Uso:  python run_local_global.py            -> results/local_vs_global.csv y results/local_global_preds/<modelo>_local.parquet
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common as C
import datos as D
import modelos_clasicos as M
import modelos_torch as T

N_LOCAL = 10
L = 336
OUT = C.ROOT / "results" / "local_global_preds"
CSV = C.ROOT / "results" / "local_vs_global.csv"
LGB_PARAMS = {"objective": "l1", "num_leaves": 63}   # misma configuración ganadora que el global


def pick_local_clients(clients: list[str]) -> list[str]:
    rng = np.random.default_rng(C.SEED)
    return sorted(rng.choice(np.array(clients), size=N_LOCAL, replace=False).tolist())


def metrics(pred: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """MAE/RMSE/WAPE/MASE de unas predicciones (unique_id, ds, y_hat) contra el panel, mismas fórmulas que common.evaluate."""
    df = pred.merge(panel[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
    assert df["y"].notna().all() and len(df) == df["unique_id"].nunique() * len(C.test_origins()) * C.H
    return C._agg(df, C.mase_scales(panel))


def local_lightgbm(cid: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    train, hist = sub[sub["ds"] <= C.TRAIN_END], sub[sub["ds"] <= C.VAL_END]
    sel = M.LGBMGlobal(LGB_PARAMS, n_estimators=2000).fit(train, valid=sub)            # nº de árboles en validación
    m = M.LGBMGlobal(LGB_PARAMS, n_estimators=max(int(sel.best_iteration), 1)).fit(hist)
    pred = C.backtest(m, sub)
    return pred, time.time() - t0


def local_torch(arch: str, cid: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    wide = D.panel_to_wide(sub)
    scaler = D.SeriesScaler().fit(wide)
    wn, ms = scaler.transform(wide), C.mase_scales(sub)
    net = T.build(arch, L)
    r = T.fit_torch(net, wn, scaler, ms, L, max_epochs=30, verbose=False)               # iteración con validación
    net = T.build(arch, L)
    T.fit_torch(net, wn, scaler, ms, L, split="trainval", fixed_epochs=r["mejor_epoca"], verbose=False)
    model = T.TorchForecaster(arch, net, scaler, L, wide.columns.tolist())
    return C.backtest(model, sub), time.time() - t0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    panel = C.load_panel()
    clients = pick_local_clients(sorted(panel["unique_id"].unique()))
    print("clientes locales:", clients, flush=True)
    sub_all = panel[panel["unique_id"].isin(clients)]

    runners = {"lightgbm": ("lightgbm_global", local_lightgbm),
               "dlinear": ("dlinear", lambda c, s: local_torch("dlinear", c, s)),
               "mlp": ("mlp", lambda c, s: local_torch("mlp", c, s))}
    rows = []
    for arch, (glob_name, fn) in runners.items():
        preds, times = [], []
        for cid in clients:
            p, dt = fn(cid, panel[panel["unique_id"] == cid])
            preds.append(p)
            times.append(dt)
            print(f"{arch} local {cid}: {dt:.0f}s", flush=True)
        local = pd.concat(preds, ignore_index=True)
        local.assign(modelo=f"{arch}_local").to_parquet(OUT / f"{arch}_local.parquet", index=False)
        glob = pd.read_parquet(C.ROOT / "results" / "preds" / f"{glob_name}.parquet")
        glob = glob[glob["unique_id"].isin(clients)][["unique_id", "ds", "y_hat"]]
        ml, mg = metrics(local, sub_all), metrics(glob, sub_all)
        # MASE por cliente para contar en cuántos clientes gana cada enfoque
        scale = C.mase_scales(sub_all)
        def per_client(pred):
            d = pred.merge(sub_all[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
            return (d["y_hat"] - d["y"]).abs().groupby(d["unique_id"]).mean() / scale
        wins = int((per_client(local) < per_client(glob)).sum())
        for kind, m, t in [("local", ml, float(np.sum(times))), ("global", mg, np.nan)]:
            rows.append({"modelo": arch, "enfoque": kind, "MAE": m["MAE"], "RMSE": m["RMSE"], "WAPE": m["WAPE"],
                         "MASE": m["MASE"], "n_clientes": N_LOCAL, "t_entrenamiento_total_s": t,
                         "clientes_donde_local_gana": wins if kind == "local" else np.nan})
        print(f"{arch}: MASE local {ml['MASE']:.3f} vs global {mg['MASE']:.3f} | local gana en {wins}/{N_LOCAL} clientes", flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)
    print(pd.DataFrame(rows).round(3).to_string())
