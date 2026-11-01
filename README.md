# Lab 9 – Pronóstico de consumo eléctrico (UCI Electricity)

Camila Richter (23183) y Marinés García (23391) · CC3092 Deep Learning · Repositorio: https://github.com/NESHGP04/Lab9-DL

Pronóstico horario a 24 h de 100 clientes (subconjunto fijo, `SEED=42`) del conjunto UCI ElectricityLoadDiagrams 2011–2014.
Protocolo único para todos los modelos: entrenamiento ≤ 2014-06-30, validación jul–sep, prueba oct–dic 2014 con 92 orígenes
diarios a las 00:00 (H = 24). Métricas: MAE, RMSE, WAPE, MAPE y MASE (escala = seasonal naive 24 h en entrenamiento).

## Resultado principal (MASE en prueba, menor es mejor)
| Modelo | MASE |
|---|---|
| Chronos-Bolt base (zero-shot) | 0.781 |
| Chronos-Bolt small (zero-shot) | 0.813 |
| LightGBM global (mejor entrenado) | 0.850 |
| iTransformer | 0.881 |
| MLP | 0.893 |
| LSTM autorregresivo | 0.900 |
| DLinear / NLinear | 0.917 / 0.928 |
| Seasonal naive 24 h | 1.088 |

Tabla completa: `results/tabla_comparativa.csv` (generada por `05_comparacion.ipynb`). Informe: `informe/Informe_Lab9.docx`.

## Estructura
- `01_datos.ipynb` – carga, DST, EDA, covariables de calendario, ventanas y auditoría de fugas.
- `02_backtest.ipynb` – backtest con origen móvil, métricas y reentrenamiento mensual.
- `03_modelos_A.ipynb` – baselines, LightGBM, MLP, DLinear, NLinear. `04_reentrenamiento_A.ipynb` – reentrenamiento mensual.
- `06_recurrentes_B.ipynb` – LSTM/GRU multi-salida, seq2seq y autorregresivo.
- `05_comparacion.ipynb` – tabla comparativa final, Pareto, MASE por h, historia, probabilísticos.
- Código: `common.py` (contrato: `load_panel`, `backtest`, `evaluate`, `save_preds`), `datos.py`, `modelos_*.py`, `comparar.py`.
- Scripts: `run_baselines.py`, `run_torch.py`, `run_reentreno.py`, `run_recurrentes.py`, `run_final_lstm_ar.py`,
  `run_neuralforecast.py` (N-HiTS, TCN, PatchTST, iTransformer, DeepAR; con puntos de control), `run_chronos.py`,
  `run_local_global.py`, `run_semillas.py`. Guía de ejecución en Colab: `COLAB_extra.md`.
- Resultados: `results/preds/<modelo>.parquet` (`unique_id, ds, modelo, y_hat`; DeepAR y Chronos añaden `q10, q50, q90`),
  `results/resumen.csv`, `results/iteraciones_*.csv`, `results/local_vs_global.csv`, `results/semillas*.csv`.

## Cómo reproducir
```bash
pip install -r requirements.txt           # NeuralForecast 1.7.x; Chronos en un entorno aparte si hay conflictos de versiones
# descargar LD2011_2014.txt (UCI) a la raíz y ejecutar 01_datos.ipynb para generar data/panel.parquet
python run_baselines.py && python run_torch.py && python run_reentreno.py
python run_neuralforecast.py --cpu nhits tcn patchtst itransformer   # DeepAR: --max-steps=1000 deepar
python run_chronos.py amazon/chronos-bolt-base
python run_local_global.py && python run_semillas.py                  # LAB9_CPU=1 fuerza CPU en PyTorch
jupyter nbconvert --to notebook --execute --inplace 05_comparacion.ipynb
```
`LD2011_2014.txt` (711 MB) y `data/panel.parquet` no se versionan.
