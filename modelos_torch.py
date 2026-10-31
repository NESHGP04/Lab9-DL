"""Modelos propios en PyTorch: MLP multi-salida, DLinear y NLinear.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import common as C
import datos as D

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_CAL = len(D.CAL_COLS)


# ------------------------------------------------------------------ redes
class NLinear(nn.Module):
    """Resta el último valor de la ventana, aplica una capa lineal L->H y lo vuelve a sumar (Zeng et al. 2023)."""
    def __init__(self, L: int, H: int = C.H):
        super().__init__()
        self.fc = nn.Linear(L, H)

    def forward(self, x, xf=None):
        last = x[:, -1:]
        return self.fc(x - last) + last


class DLinear(nn.Module):
    """Descompone en tendencia (media móvil) y resto; una capa lineal L->H por componente y se suman."""
    def __init__(self, L: int, H: int = C.H, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.fc_trend, self.fc_seas = nn.Linear(L, H), nn.Linear(L, H)

    def forward(self, x, xf=None):
        pad = (self.kernel - 1) // 2
        xp = torch.cat([x[:, :1].expand(-1, pad), x, x[:, -1:].expand(-1, pad)], dim=1)   # relleno replicado
        trend = xp.unfold(1, self.kernel, 1).mean(-1)
        return self.fc_trend(trend) + self.fc_seas(x - trend)


class MLP(nn.Module):
    """MLP multi-salida: ventana de consumo (L) + calendario futuro aplanado (H*F) -> H pasos."""
    def __init__(self, L: int, H: int = C.H, hidden: int = 512, layers: int = 2, dropout: float = 0.1,
                 covariates: bool = True):
        super().__init__()
        self.covariates = covariates
        d_in = L + (H * N_CAL if covariates else 0)
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d_in = hidden
        self.body, self.head = nn.Sequential(*blocks), nn.Linear(d_in, H)

    def forward(self, x, xf=None):
        z = torch.cat([x, xf.flatten(1)], dim=1) if self.covariates else x
        return self.head(self.body(z))


def build(name: str, L: int, **kw) -> nn.Module:
    return {"mlp": MLP, "dlinear": DLinear, "nlinear": NLinear}[name](L, **kw)


def count_params(net: nn.Module) -> tuple[int, int]:
    return (sum(p.numel() for p in net.parameters()),
            sum(p.numel() for p in net.parameters() if p.requires_grad))


# ------------------------------------------------------------------ datos en memoria (batches vectorizados)
class Windows:
    """Mismas ventanas que datos.WindowDataset, pero indexadas en bloque sobre tensores en `device`
    (evita el __getitem__ de Python: ~2 M de ventanas por época)."""

    def __init__(self, wn: pd.DataFrame, split: str, L: int, train_start: pd.Timestamp | None = None,
                 train_end: pd.Timestamp | None = None):
        ds = D.WindowDataset(wn, split, L, C.H, train_start=train_start, train_end=train_end)
        self.L, self.n_series, self.series = L, ds.n_series, ds.series
        self.values, self.cal = ds.values.to(DEVICE), ds.cal.to(DEVICE)
        self.starts = torch.tensor(ds.starts, device=DEVICE)
        self.ar_in = torch.arange(-L, 0, device=DEVICE)
        self.ar_out = torch.arange(C.H, device=DEVICE)

    def __len__(self):
        return self.n_series * len(self.starts)

    def get(self, idx: torch.Tensor):
        i, s = idx // len(self.starts), self.starts[idx % len(self.starts)]
        x = self.values[i[:, None], s[:, None] + self.ar_in]
        y = self.values[i[:, None], s[:, None] + self.ar_out]
        xf = self.cal[s[:, None] + self.ar_out]
        return x, xf, y, i


# ------------------------------------------------------------------ entrenamiento
def evaluate_windows(net, win: Windows, scaler: D.SeriesScaler, mase_scale: pd.Series, bs: int = 16) -> dict:
    """Pérdida L1 normalizada, MAE (kWh) y MASE sobre todas las ventanas de `win`."""
    net.eval()
    abs_sum = torch.zeros(win.n_series, device=DEVICE)
    with torch.no_grad():
        for k in range(0, len(win), bs):
            x, xf, y, i = win.get(torch.arange(k, min(k + bs, len(win)), device=DEVICE))
            abs_sum.index_add_(0, i, (net(x, xf) - y).abs().sum(1))
    mae_norm_i = (abs_sum / (len(win.starts) * C.H)).cpu().numpy()
    std = scaler.std_.reindex(win.series).to_numpy()
    mae_i = mae_norm_i * std
    return {"val_loss": float(mae_norm_i.mean()), "val_MAE": float(mae_i.mean()),
            "val_MASE": float(np.mean(mae_i / mase_scale.reindex(win.series).to_numpy()))}


def fit_torch(net, wn: pd.DataFrame, scaler: D.SeriesScaler, mase_scale: pd.Series, L: int, *,
              split: str = "train", train_start: pd.Timestamp | None = None,
              train_end: pd.Timestamp | None = None, max_epochs: int = 30,
              fixed_epochs: int | None = None, patience: int = 4, bs: int = 16, lr: float = 1e-3,
              weight_decay: float = 0.0, seed: int = C.SEED, verbose: bool = True) -> dict:
    """Entrena con L1 (la métrica es MAE/MASE). Con split="train" valida en cada época y restaura la mejor
    (early stopping sobre VALIDACIÓN, nunca sobre la prueba). Con split="trainval" entrena `fixed_epochs`
    épocas con datos hasta 2014-09-30 y no valida (la validación ya es parte del entrenamiento)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net.to(DEVICE)
    tr = Windows(wn, split, L, train_start, train_end)
    va = Windows(wn, "val", L) if split == "train" else None
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    n_epochs = fixed_epochs or max_epochs
    log, best, best_state, bad = [], np.inf, None, 0
    t_total = time.time()
    for ep in range(1, n_epochs + 1):
        net.train()
        t0, tot, nb = time.time(), torch.zeros((), device=DEVICE), 0
        perm = torch.randperm(len(tr), device=DEVICE)
        for k in range(0, len(tr), bs):
            x, xf, y, _ = tr.get(perm[k:k + bs])
            loss = (net(x, xf) - y).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()
            nb += 1
        row = {"epoca": ep, "train_loss": float(tot) / nb}
        if va is not None:
            row.update(evaluate_windows(net, va, scaler, mase_scale))
        if DEVICE.type == "mps":
            torch.mps.synchronize()
            row["mem_gpu_MB"] = torch.mps.driver_allocated_memory() / 2 ** 20
        row["t_epoca_s"] = time.time() - t0
        log.append(row)
        if verbose:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        if va is not None:
            if row["val_loss"] < best - 1e-5:
                best, bad = row["val_loss"], 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    best_ep = int(np.argmin([r["val_loss"] for r in log])) + 1 if va is not None else n_epochs
    return {"log": log, "mejor_epoca": best_ep, "t_total_s": time.time() - t_total}


# ------------------------------------------------------------------ envoltorio para backtest
class TorchForecaster(C.ForecastModel):
    """Convierte una red entrenada en un ForecastModel: arma x (últimas L h) y el calendario futuro desde `history`."""

    def __init__(self, name: str, net: nn.Module, scaler: D.SeriesScaler, L: int, series: list[str]):
        self.name, self.net, self.scaler, self.L, self.series = name, net.to(DEVICE).eval(), scaler, L, series
        self.infer_time = 0.0

    def fit(self, train):  # el entrenamiento se hace antes con fit_torch
        return self

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        t0 = time.time()
        tail = history[history["ds"] >= origin - pd.Timedelta(hours=self.L)]
        wide = D.panel_to_wide(tail).reindex(columns=self.series)
        assert len(wide) == self.L and wide.index[-1] == origin - pd.Timedelta(hours=1)
        x = torch.tensor(self.scaler.transform(wide).to_numpy(dtype=np.float32).T, device=DEVICE)   # (N, L)
        ds = pd.date_range(origin, periods=C.H, freq="h")
        cal = torch.tensor(D.calendar_features(ds).to_numpy(dtype=np.float32), device=DEVICE)
        with torch.no_grad():
            yn = self.net(x, cal[None].expand(len(self.series), -1, -1)).cpu().numpy()
        y = yn * self.scaler.std_.reindex(self.series).to_numpy()[:, None] \
            + self.scaler.mean_.reindex(self.series).to_numpy()[:, None]
        self.infer_time += time.time() - t0
        return pd.DataFrame({"unique_id": np.repeat(self.series, C.H), "ds": np.tile(ds, len(self.series)),
                             "y_hat": np.clip(y.reshape(-1), 0, None)})


class RetrainableTorch(C.ForecastModel):
    """Reentrenamiento con ventana expansiva (backtest(..., retrain="monthly")): en cada fit(history) crea la red desde
    cero y la entrena `epochs` épocas con todas las ventanas cuyo objetivo cae antes del origen. El escalador no cambia
    (estadísticos de entrenamiento), para aislar el efecto de reentrenar los pesos."""

    def __init__(self, name: str, arch: str, L: int, epochs: int, wn: pd.DataFrame, scaler: D.SeriesScaler,
                 mase_scale: pd.Series, series: list[str], **kw):
        self.name, self.arch, self.L, self.epochs, self.kw = name, arch, L, epochs, kw
        self.wn, self.scaler, self.mase_scale, self.series = wn, scaler, mase_scale, series
        self.fit_times, self.fit_ends, self.inner = [], [], None

    def fit(self, train: pd.DataFrame) -> "RetrainableTorch":
        end = train["ds"].max()
        net = build(self.arch, self.L, **self.kw)
        r = fit_torch(net, self.wn, self.scaler, self.mase_scale, self.L, split="trainval", train_end=end,
                      fixed_epochs=self.epochs, verbose=False)
        self.fit_times.append(r["t_total_s"]); self.fit_ends.append(end)
        self.inner = TorchForecaster(self.name, net, self.scaler, self.L, self.series)
        return self

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        return self.inner.predict(history, origin)
