"""Modelos recurrentes: LSTM/GRU multi-salida y seq2seq con calendario en el decoder.

Misma interfaz que modelos_torch.py: forward(x, xf) con
    x  : (B, L)    consumo normalizado de las últimas L horas
    xf : (B, H, F) calendario conocido de las H horas a pronosticar
    ->   (B, H)    consumo normalizado pronosticado
Así se entrenan con modelos_torch.fit_torch y se evalúan con TorchForecaster + common.backtest.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import common as C
import datos as D

N_CAL = len(D.CAL_COLS)
_CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


class RNNMulti(nn.Module):
    """Multi-salida directa: la RNN lee la ventana (L pasos, 1 canal); el último estado oculto
    (+ calendario futuro aplanado) pasa por una cabeza que emite las H horas de golpe."""

    def __init__(self, L: int, H: int = C.H, cell: str = "lstm", hidden: int = 64, layers: int = 1,
                 dropout: float = 0.1, covariates: bool = True):
        super().__init__()
        self.covariates = covariates
        self.rnn = _CELLS[cell](1, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
        d = hidden + (H * N_CAL if covariates else 0)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, H))

    def forward(self, x, xf=None):
        out, _ = self.rnn(x.unsqueeze(-1))
        z = out[:, -1]
        if self.covariates:
            z = torch.cat([z, xf.flatten(1)], dim=1)
        return self.head(z)


class Seq2Seq(nn.Module):
    """Encoder-decoder: el encoder resume la ventana en su estado final; el decoder arranca con ese estado
    y recorre las H horas futuras recibiendo SOLO el calendario de cada hora (no realimenta sus predicciones,
    por lo que no necesita teacher forcing y es compatible con fit_torch). Una capa lineal convierte cada
    paso del decoder en un valor."""

    def __init__(self, L: int, H: int = C.H, cell: str = "gru", hidden: int = 64, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        drop = dropout if layers > 1 else 0.0
        self.enc = _CELLS[cell](1, hidden, num_layers=layers, batch_first=True, dropout=drop)
        self.dec = _CELLS[cell](N_CAL, hidden, num_layers=layers, batch_first=True, dropout=drop)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, xf):
        _, state = self.enc(x.unsqueeze(-1))     # state = h (GRU) o (h, c) (LSTM)
        o, _ = self.dec(xf.contiguous(), state)  # (B, H, hidden); contiguous: en inferencia xf viene de un expand()
        return self.out(o).squeeze(-1)


NAMES = {
    "lstm_multi": (RNNMulti, dict(cell="lstm")),
    "gru_multi": (RNNMulti, dict(cell="gru")),
    "seq2seq_lstm": (Seq2Seq, dict(cell="lstm")),
    "seq2seq_gru": (Seq2Seq, dict(cell="gru")),
}


def build_rnn(name: str, L: int, **kw) -> nn.Module:
    cls, base = NAMES[name]
    return cls(L, **{**base, **kw})