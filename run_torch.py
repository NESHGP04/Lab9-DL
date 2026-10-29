from __future__ import annotations

import math
import subprocess
import sys
import time

import pandas as pd

import common as C
import datos as D
import modelos_torch as T

HW = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
HW = f"{HW} (GPU MPS)" if T.DEVICE.type == "mps" else HW
MODELS = {"mlp": dict(), "dlinear": dict(), "nlinear": dict()}
LS = (48, 168, 336)

if __name__ == "__main__":
    panel = C.load_panel()
    wide = D.panel_to_wide(panel)
    scaler = D.SeriesScaler().fit(wide)               # estadísticos de entrenamiento (<= 2014-06-30)
    wn = scaler.transform(wide)
    mase_scale = C.mase_scales(panel)
    series = wide.columns.tolist()

    only_hist = "hist" in sys.argv[1:]
    logs, summ = [], []
    # 1) iteraciones sobre validación
    for name, kw in ([] if only_hist else MODELS.items()):
        for L in LS:
            net = T.build(name, L, **kw)
            tot, trn = T.count_params(net)
            print(f"== {name} L={L} params={tot}", flush=True)
            r = T.fit_torch(net, wn, scaler, mase_scale, L, max_epochs=25)
            for row in r["log"]:
                logs.append({"modelo": name, "L": L, "params": tot, **row})
            b = r["log"][r["mejor_epoca"] - 1]
            summ.append({"modelo": name, "L": L, "params_totales": tot, "params_entrenables": trn,
                         "mejor_epoca": r["mejor_epoca"], "epocas_corridas": len(r["log"]),
                         "val_MAE": b["val_MAE"], "val_MASE": b["val_MASE"], "val_loss": b["val_loss"],
                         "t_total_s": r["t_total_s"], "t_epoca_s": r["t_total_s"] / len(r["log"]),
                         "mem_gpu_MB": b.get("mem_gpu_MB")})
            pd.DataFrame(logs).to_csv(C.ROOT / "results" / "iteraciones_A.csv", index=False)
            pd.DataFrame(summ).to_csv(C.ROOT / "results" / "iteraciones_A_resumen.csv", index=False)
    summ = pd.read_csv(C.ROOT / "results" / "iteraciones_A_resumen.csv") if only_hist else pd.DataFrame(summ)

    # 2) y 3) modelos finales y variantes de historia
    for name, kw in MODELS.items():
        best = summ[summ.modelo == name].sort_values("val_MASE").iloc[0]
        L, n_ep = int(best["L"]), int(best["mejor_epoca"])
        print(f"### {name}: mejor L={L}, {n_ep} épocas (val MASE {best['val_MASE']:.4f})", flush=True)
        runs = [(name, None), (f"{name}_hist03m", 3), (f"{name}_hist12m", 12), (f"{name}_hist30m", 30)]
        for tag, months in (runs[1:] if only_hist else runs):
            start = None if months is None else C.VAL_END - pd.DateOffset(months=months) + pd.Timedelta(hours=1)
            # con menos historia hay menos ventanas por época: se escalan las épocas para igualar los pasos de gradiente
            ep = n_ep if months is None else math.ceil(n_ep * 33 / months)
            net = T.build(name, L, **kw)
            r = T.fit_torch(net, wn, scaler, mase_scale, L, split="trainval", train_start=start,
                            fixed_epochs=ep, verbose=False)
            model = T.TorchForecaster(tag, net, scaler, L, series)
            C.save_preds(C.backtest(model, panel), tag)
            tot, trn = T.count_params(net)
            C.log_resumen(tag, familia={"mlp": "MLP", "dlinear": "Lineal", "nlinear": "Lineal"}[name], L_horas=L,
                          estrategia="multi-salida", covariables="sí" if name == "mlp" else "no",
                          params_totales=tot, params_entrenables=trn, t_entrenamiento_s=round(r["t_total_s"], 1),
                          t_epoca_s=round(r["t_total_s"] / ep, 1), t_inferencia_s=round(model.infer_time, 1),
                          mem_gpu_MB=round(r["log"][-1].get("mem_gpu_MB", float("nan")), 0), hardware=HW,
                          meses_historia=months or 33, config=f"L={L}, {ep} épocas, Adam lr=1e-3, L1")
            print(tag, f"entreno {r['t_total_s']:.0f}s", flush=True)
