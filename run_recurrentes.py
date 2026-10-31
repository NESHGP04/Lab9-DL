"""Persona B – B3: entrena LSTM/GRU multi-salida y seq2seq, con el MISMO protocolo que run_torch.py (A):
  1) iteración con early stopping sobre VALIDACIÓN (log por época),
  2) reentrenamiento final con datos hasta 2014-09-30 por `mejor_epoca` épocas,
  3) backtest de 92 orígenes -> results/preds/<modelo>.parquet + fila en results/resumen.csv.

Uso:
  python run_recurrentes.py --probe            # 1 época con 10 clientes: mide tiempo (no escribe resultados)
  python run_recurrentes.py                    # los 4 modelos
  python run_recurrentes.py gru_multi seq2seq_gru   # solo algunos
"""
from __future__ import annotations

import subprocess
import sys

import pandas as pd
import torch

import common as C
import datos as D
import modelos_recurrentes as R
import modelos_torch as T

L = 168


def pick_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        return torch.device("mps"), f"{cpu} (GPU MPS)"
    return torch.device("cpu"), "CPU"


def mem_mb(r: dict) -> float:
    if T.DEVICE.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2 ** 20, 0)
    return round(r["log"][-1].get("mem_gpu_MB", float("nan")), 0)


if __name__ == "__main__":
    # modelos_torch fija DEVICE al importarse (mps o cpu); aquí se sobreescribe para poder usar CUDA (Colab).
    T.DEVICE, HW = pick_device()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    probe = "--probe" in sys.argv
    names = args or list(R.NAMES)

    panel = C.load_panel()
    wide = D.panel_to_wide(panel)
    scaler = D.SeriesScaler().fit(wide)
    wn = scaler.transform(wide)
    mase_scale = C.mase_scales(panel)
    series = wide.columns.tolist()
    print(f"dispositivo: {T.DEVICE} | {HW}", flush=True)

    if probe:
        wn_p = wn.iloc[:, :10]
        for name in names:
            net = R.build_rnn(name, L)
            r = T.fit_torch(net, wn_p, scaler, mase_scale, L, max_epochs=1, patience=1, verbose=False)
            row = r["log"][0]
            print(f"{name}: params={T.count_params(net)[0]}, {row['t_epoca_s']:.1f} s/época (10 clientes) -> "
                  f"~{row['t_epoca_s'] * len(series) / 10:.0f} s/época con {len(series)} clientes | "
                  f"val_MASE={row['val_MASE']:.3f}", flush=True)
        sys.exit()

    log_path = C.ROOT / "results" / "iteraciones_B.csv"
    summary_path = C.ROOT / "results" / "iteraciones_B_resumen.csv"

    # Conserva los modelos anteriores y reemplaza únicamente
    # los registros de los modelos que se ejecutarán ahora.
    def cargar_otros(path):
        if not path.exists():
            return []

        df = pd.read_csv(path)
        return df.loc[~df["modelo"].isin(names)].to_dict("records")

    logs = cargar_otros(log_path)
    summ = cargar_otros(summary_path)
    
    for name in names:
        # 1) iteración sobre validación
        if T.DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        net = R.build_rnn(name, L)
        tot, trn = T.count_params(net)
        print(f"== {name} L={L} params={tot}", flush=True)
        r = T.fit_torch(net, wn, scaler, mase_scale, L, max_epochs=15, patience=3)
        for row in r["log"]:
            logs.append({"modelo": name, "L": L, "params": tot, **row})
        b = r["log"][r["mejor_epoca"] - 1]
        summ.append({"modelo": name, "L": L, "params_totales": tot, "params_entrenables": trn,
                     "mejor_epoca": r["mejor_epoca"], "epocas_corridas": len(r["log"]),
                     "val_MAE": b["val_MAE"], "val_MASE": b["val_MASE"], "val_loss": b["val_loss"],
                     "t_total_s": r["t_total_s"], "t_epoca_s": r["t_total_s"] / len(r["log"]),
                     "mem_gpu_MB": mem_mb(r)})
        pd.DataFrame(logs).to_csv(C.ROOT / "results" / "iteraciones_B.csv", index=False)
        pd.DataFrame(summ).to_csv(C.ROOT / "results" / "iteraciones_B_resumen.csv", index=False)

        # 2) modelo final (entrena hasta 2014-09-30, sin validar) y 3) backtest
        n_ep = r["mejor_epoca"]
        net = R.build_rnn(name, L)
        rf = T.fit_torch(net, wn, scaler, mase_scale, L, split="trainval", fixed_epochs=n_ep, verbose=False)
        model = T.TorchForecaster(name, net, scaler, L, series)
        C.save_preds(C.backtest(model, panel), name)
        C.log_resumen(name, familia="Recurrente", L_horas=L,
                      estrategia="multi-salida" if name.endswith("multi") else "encoder-decoder (calendario en decoder)",
                      covariables="sí", params_totales=tot, params_entrenables=trn,
                      t_entrenamiento_s=round(rf["t_total_s"], 1), t_epoca_s=round(rf["t_total_s"] / n_ep, 1),
                      t_inferencia_s=round(model.infer_time, 1), mem_gpu_MB=mem_mb(rf), hardware=HW,
                      meses_historia=33, config=f"L={L}, {n_ep} épocas, Adam lr=1e-3, L1, hidden=64")
        print(name, f"final {rf['t_total_s']:.0f}s", flush=True)