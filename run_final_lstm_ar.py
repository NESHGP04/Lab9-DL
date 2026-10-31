"""Entrenamiento final recuperable de lstm_ar, sin repetir la selección.

Colocar junto a common.py, datos.py, modelos_torch.py y modelos_recurrentes.py.
Ejemplo (Drive previamente montado):
  python -u run_final_lstm_ar.py --save-dir /content/drive/MyDrive/Lab9-DL_recuperacion

Repetir el mismo comando reanuda desde la última época guardada.
Si se interrumpe una época, se repite SOLO esa época.
--check-only valida configuración y datos sin entrenar.
--epochs-this-run 1 ejecuta una época pendiente y sale guardando el avance.
No ejecutar dos instancias contra la misma carpeta de checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

import common as C
import datos as D
import modelos_recurrentes as R
import modelos_torch as T

NAME, L, EPOCHS, BS = "lstm_ar", 168, 15, 128

# Transcripción de la salida real proporcionada por Camila. No son nuevas ejecuciones.
# epoca, train_loss, val_loss, val_MAE, val_MASE, t_epoca_s
SELECTION = [
    (1, .2892, .2649, 44.3591, 1.0869, 583.2832),
    (2, .2516, .2519, 41.6167, 1.0375, 591.3721),
    (3, .2371, .2378, 39.5576, .9817, 602.113),
    (4, .2273, .2285, 37.975, .9445, 591.4088),
    (5, .2266, .2278, 38.7361, .9471, 591.2199),
    (6, .218, .2249, 37.5181, .9271, 594.1488),
    (7, .2152, .2215, 37.5327, .9153, 597.4105),
    (8, .2127, .2205, 37.0536, .9063, 599.6729),
    (9, .2105, .2203, 37.444, .9092, 598.8707),
    (10, .217, .2162, 36.4428, .89, 596.265),
    (11, .2083, .217, 36.8127, .893, 595.1946),
    (12, .2065, .2141, 36.248, .88, 597.1219),
    (13, .2052, .2146, 36.8978, .8861, 601.2816),
    (14, .2057, .2204, 36.53, .8939, 602.3508),
    (15, .206, .2135, 36.2426, .8812, 599.415),
]


def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def merge_model_rows(path, rows):
    """Reemplaza solo lstm_ar; conserva las filas de los demás modelos."""
    path = Path(path)
    other = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not other.empty:
        if "modelo" not in other:
            raise ValueError(f"Falta columna modelo en {path}")
        other = other.loc[other.modelo != NAME]
    atomic_csv(pd.concat([other, rows], ignore_index=True), path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, payload):
    """Publica un checkpoint completo y conserva el anterior como respaldo."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    # Verifica que el archivo es legible antes de sustituir el anterior.
    torch.load(tmp, map_location="cpu", weights_only=False)
    if path.exists():
        backup_tmp = path.with_suffix(".previous.tmp")
        shutil.copy2(path, backup_tmp)
        os.replace(backup_tmp, path.with_suffix(".previous.pt"))
    os.replace(tmp, path)


def load_checkpoint(path):
    """Carga exclusivamente checkpoints propios generados por este script."""
    path = Path(path)
    candidates = [path, path.with_suffix(".previous.pt")]
    found = False
    for candidate in candidates:
        if not candidate.exists():
            continue
        found = True
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if not {"config", "epoch", "model", "optimizer", "rng_cpu", "log"} <= state.keys():
                raise ValueError("Checkpoint incompleto")
        except Exception as exc:
            print(f"No se pudo leer {candidate.name}: {exc}", flush=True)
            continue
        print(f"Checkpoint cargado: {candidate.name}, época {state['epoch']}", flush=True)
        # Si el principal estaba dañado, conserva el respaldo válido.
        if candidate != path and path.exists():
            os.replace(path, path.with_suffix(".damaged"))
        return state
    if found:
        raise RuntimeError("No hay un checkpoint legible. No se reiniciará silenciosamente.")
    return None


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--epochs-this-run", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs_this_run < 1:
        parser.error("--epochs-this-run debe ser positivo")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. Activa la GPU antes de entrenar.")
    if str(args.save_dir).startswith("/content/drive/") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Primero monta Drive con drive.mount('/content/drive').")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    # Comprueba escritura antes de invertir tiempo en el entrenamiento.
    probe = args.save_dir / ".write_probe"
    probe.write_text("ok")
    probe.unlink()
    T.DEVICE = torch.device(args.device)
    hardware = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print(f"Dispositivo: {T.DEVICE} | {hardware}", flush=True)
    print(f"Destino persistente: {args.save_dir}", flush=True)

    panel = C.load_panel()
    wide = D.panel_to_wide(panel)
    if wide.shape[1] != C.N_CLIENTES or wide.isna().any().any():
        raise ValueError("Panel incompleto o número de clientes distinto del contrato")
    scaler = D.SeriesScaler().fit(wide)  # Estadísticos SOLO de train.
    wn = scaler.transform(wide)
    config = dict(modelo=NAME, L=L, H=C.H, epochs=EPOCHS, bs=BS, lr=.001,
                  seed=C.SEED, split="trainval", train_end=str(C.VAL_END),
                  series=list(wide.columns), panel_sha256=file_hash(C.PANEL_PATH),
                  model_sha256=file_hash(R.__file__), torch_version=str(torch.__version__))
    checkpoint = args.save_dir / "lstm_ar_final.pt"
    state = load_checkpoint(checkpoint)
    if state is not None and state["config"] != config:
        raise ValueError("Los datos, modelo, versión de PyTorch o configuración cambiaron. "
                         "Usa los mismos archivos para reanudar; no se alteró el checkpoint.")
    print(f"Configuración: L={L}, H={C.H}, batch={BS}, clientes={wide.shape[1]}, "
          f"15 épocas finales; selección omitida.", flush=True)
    if args.check_only:
        print("Comprobación correcta. No se entrenó ninguna época.", flush=True)
        return

    torch.manual_seed(C.SEED)  # Inicialización nueva y reproducible, no pesos recuperados.
    np.random.seed(C.SEED)
    net = R.build_rnn(NAME, L).to(T.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    windows = T.Windows(wn, "trainval", L)
    log, completed = [], 0
    if state is not None:
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        log, completed = state["log"], state["epoch"]
        torch.set_rng_state(state["rng_cpu"])
        if args.device == "cuda" and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        if "rng_numpy" in state:
            np.random.set_state(state["rng_numpy"])
    if not 0 <= completed <= EPOCHS or len(log) != completed:
        raise ValueError("Número de épocas inconsistente en checkpoint")
    if state is None:
        # Comprueba también la serialización antes de la primera época larga.
        save_checkpoint(checkpoint, dict(
            config=config, epoch=0, log=[],
            model=cpu_tree(net.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
            rng_numpy=np.random.get_state(),
        ))
        print("Checkpoint inicial guardado correctamente.", flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop = min(EPOCHS, completed + args.epochs_this_run)
    print(f"Épocas ya guardadas: {completed}/{EPOCHS}", flush=True)

    for epoch in range(completed + 1, stop + 1):
        net.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(windows), device=T.DEVICE)
        total = torch.zeros((), device=T.DEVICE)
        batches = (len(windows) + BS - 1) // BS
        for batch, start in enumerate(range(0, len(windows), BS), 1):
            x, xf, y, _ = windows.get(perm[start:start + BS])
            optimizer.zero_grad(set_to_none=True)
            loss = (net(x, xf) - y).abs().mean()
            loss.backward()
            optimizer.step()
            total += loss.detach()
            if batch % 2000 == 0:
                print(f"Época {epoch}/{EPOCHS}: batch {batch}/{batches}", flush=True)
        if args.device == "cuda":
            torch.cuda.synchronize()
        row = dict(epoca=epoch, train_loss=float(total) / batches,
                   t_epoca_s=time.perf_counter() - t0, hardware=hardware,
                   mem_gpu_MB=(torch.cuda.max_memory_allocated() / 2**20
                               if args.device == "cuda" else float("nan")))
        log.append(row)
        save_checkpoint(checkpoint, dict(
            config=config, epoch=epoch, log=log,
            model=cpu_tree(net.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
            rng_numpy=np.random.get_state(),
        ))
        atomic_csv(pd.DataFrame(log), args.save_dir / "entrenamiento_final_lstm_ar.csv")
        print(f"Época {epoch}/{EPOCHS} GUARDADA EN DRIVE | "
              f"loss={row['train_loss']:.4f} | {row['t_epoca_s']:.1f}s", flush=True)
    if stop < EPOCHS:
        print("Avance guardado. Repite el mismo comando para continuar.", flush=True)
        return

    # Ya hay un checkpoint final: si falla el backtest se puede repetir SIN entrenar.
    print("15/15 épocas guardadas. Iniciando backtest de 92 orígenes...", flush=True)
    model = T.TorchForecaster(NAME, net, scaler, L, list(wide.columns))
    preds = C.backtest(model, panel, verbose=True)
    local_preds = C.save_preds(preds, NAME)
    persistent_results = args.save_dir / "results"
    (persistent_results / "preds").mkdir(parents=True, exist_ok=True)
    destination = persistent_results / "preds" / local_preds.name
    tmp = destination.with_suffix(".tmp")
    shutil.copy2(local_preds, tmp)
    os.replace(tmp, destination)

    tot, trn = T.count_params(net)
    elapsed = sum(r["t_epoca_s"] for r in log)
    memory = max((r["mem_gpu_MB"] for r in log if pd.notna(r["mem_gpu_MB"])), default=float("nan"))
    C.log_resumen(NAME, familia="Recurrente", L_horas=L,
                  estrategia="autoregresiva, sin teacher forcing", covariables="sí",
                  params_totales=tot, params_entrenables=trn,
                  t_entrenamiento_s=round(elapsed, 1), t_epoca_s=round(elapsed / EPOCHS, 1),
                  t_inferencia_s=round(model.infer_time, 1), mem_gpu_MB=memory,
                  hardware="; ".join(dict.fromkeys(r["hardware"] for r in log)),
                  meses_historia=33,
                  config="L=168, H=24, 15 épocas finales, Adam lr=1e-3, L1, "
                         "hidden=64, bs=128, selección previa por val_loss; "
                         "tiempo suma épocas completadas, excluye intentos interrumpidos y checkpoints")
    selection = pd.DataFrame(SELECTION, columns=[
        "epoca", "train_loss", "val_loss", "val_MAE", "val_MASE", "t_epoca_s"])
    selection = selection.assign(modelo=NAME, L=L, params=tot, bs=BS,
                                 fuente="transcripción de salida original, redondeada")
    best = selection.loc[selection.val_loss.idxmin()]
    summary = pd.DataFrame([dict(
        modelo=NAME, L=L, params_totales=tot, params_entrenables=trn,
        mejor_epoca=15, epocas_corridas=15, val_MAE=best.val_MAE,
        val_MASE=best.val_MASE, val_loss=best.val_loss,
        t_total_s=selection.t_epoca_s.sum(), t_epoca_s=selection.t_epoca_s.mean(),
        mem_gpu_MB=float("nan"), bs=BS,
        fuente="transcripción de salida original, redondeada")])
    for root in [C.ROOT / "results", persistent_results]:
        merge_model_rows(root / "iteraciones_B.csv", selection)
        merge_model_rows(root / "iteraciones_B_resumen.csv", summary)
    own_summary = pd.read_csv(C.RESUMEN_PATH)
    merge_model_rows(persistent_results / "resumen.csv", own_summary.loc[own_summary.modelo == NAME])
    result = C.evaluate(local_preds, panel)
    metrics = pd.DataFrame([dict(modelo=NAME, **result["global"],
                                 MASE_22_28dic=result["xmas"]["MASE"], pronosticos=result["n"])])
    atomic_csv(metrics, persistent_results / "metricas_lstm_ar.csv")
    print(metrics.to_string(index=False), flush=True)
    print(f"lstm_ar final. Pronósticos y resultados guardados en {persistent_results}", flush=True)


if __name__ == "__main__":
    main()
