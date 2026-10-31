# Borrador de las secciones IV–XI (resultados de A + redes recurrentes de B)

> Ya integrados los 5 recurrentes de B (LSTM/GRU multi-salida, seq2seq y LSTM autorregresivo). Los `[PENDIENTE B]` que quedan corresponden a NeuralForecast, DeepAR, Chronos, local vs global y semillas; se llenan al subir B esos modelos y re-ejecutar `05_comparacion.ipynb`. Límite del PDF: 5 páginas, así que las tablas están compactas; el detalle por época va en el anexo / CSV.

---

## IV. Tabla de iteraciones

Las iteraciones se corren sobre **validación** (jul–sep 2014); la prueba se usa una sola vez por modelo. Entrenamiento con pérdida L1 sobre consumo normalizado (z-score por serie), Adam (lr 1e-3, batch 1024), early stopping con paciencia 4 sobre la pérdida de validación, H = 24, semilla 42, historia de entrenamiento = todos los datos hasta 2014-06-30 (30 meses). Se varió la ventana **L** (48, 168, 336 h) para el MLP, DLinear y NLinear, y la **historia** (sección siguiente).

| Modelo | L (h) | Parámetros | Época mejor / corridas | MAE val (kWh) | MASE val | t/época (s) | t total (s) |
|---|---|---|---|---|---|---|---|
| MLP | 48 | 386 072 | 21 / 25 | 35.28 | 0.868 | 14.5 | 361 |
| MLP | 168 | 447 512 | 8 / 12 | 32.23 | 0.799 | 16.0 | 192 |
| **MLP** | **336** | 533 528 | 9 / 13 | 32.02 | **0.795** | 17.2 | 223 |
| DLinear | 48 | 2 352 | 4 / 8 | 37.00 | 0.912 | 5.7 | 45 |
| DLinear | 168 | 8 112 | 4 / 8 | 33.09 | 0.835 | 6.3 | 51 |
| **DLinear** | **336** | 16 176 | 3 / 7 | 32.63 | **0.824** | 7.8 | 54 |
| NLinear | 48 | 1 176 | 4 / 8 | 36.97 | 0.912 | 3.2 | 26 |
| NLinear | 168 | 4 056 | 4 / 8 | 33.01 | 0.833 | 4.5 | 36 |
| **NLinear** | **336** | 8 088 | 2 / 6 | 32.58 | **0.822** | 5.4 | 32 |

Hardware: Apple M5, GPU (MPS). La memoria de GPU reportada por el driver fue ≈1.1–1.2 GB en todas las corridas (es una cifra acumulada del proceso, no el pico individual de cada modelo). El MAE y MASE de validación por época, junto con las pérdidas de entrenamiento y validación, están en `results/iteraciones_A.csv`.

**LightGBM** (selección en validación, con early stopping; 63–127 hojas, lr 0.05, rezagos 1–24 y 168 + calendario):

| Configuración | Objetivo | Hojas | Árboles | MAE val (norm.) | t (s) |
|---|---|---|---|---|---|
| **l1_h63** | L1 | 63 | 2 000 (tope) | **0.1965** | 210 |
| l2_h127 | L2 | 127 | 1 618 | 0.1975 | 215 |
| l2_h63 | L2 | 63 | 1 750 | 0.1984 | 154 |
| l2_h31 | L2 | 31 | 2 000 (tope) | 0.1996 | 105 |

**Efecto de la historia de entrenamiento** (MASE en prueba, modelo reentrenado con los últimos *k* meses antes de 2014-09-30; épocas escaladas para igualar pasos de gradiente, árboles elegidos en validación):

| Meses de historia | LightGBM | MLP | DLinear | NLinear |
|---|---|---|---|---|
| 3 | 1.005 | 1.413 | 0.915 | 0.908 |
| 12 | 0.853 | 0.849 | 0.904 | 0.903 |
| 30 | 0.851 | 0.857 | 0.923 | 0.912 |
| 33 (todo) | 0.850 | 0.893 | 0.917 | 0.928 |

**Redes recurrentes (B)**, misma pérdida, optimizador y validación; una capa, estado oculto 64, dropout 0.1, L = 168, 15 épocas (tope), paciencia 3, hardware Tesla T4. El LSTM autorregresivo se entrenó con batch 128 (los demás, 1024) y los valores de validación de sus épocas se transcribieron de la salida de Colab, redondeados:

| Modelo | L (h) | Parámetros | Época mejor / corridas | MAE val (kWh) | MASE val | t/época (s) | t total (s) |
|---|---|---|---|---|---|---|---|
| LSTM multi-salida | 168 | 33 624 | 15 / 15 | 41.04 | 0.994 | 30.5 | 458 |
| GRU multi-salida | 168 | 29 336 | 15 / 15 | 41.15 | 0.997 | 31.6 | 474 |
| LSTM seq2seq | 168 | 35 905 | 15 / 15 | 37.82 | 0.927 | 36.7 | 550 |
| **GRU seq2seq** | 168 | 26 945 | 15 / 15 | 36.89 | **0.918** | 37.4 | 561 |
| LSTM autorregresivo (bs 128) | 168 | 21 825 | 15 / 15 | 36.24 | 0.881 | 596 | 8 941 |

La mejor época fue la última (15) en los cinco: la pérdida de validación seguía bajando (p. ej. GRU seq2seq 0.304 → 0.227; LSTM autorregresivo 0.265 → 0.214), es decir, **no convergieron** y su resultado es una cota superior del error alcanzable. B solo probó una configuración (L = 168, sin variar L, normalización ni historia), así que no hay tabla de iteraciones comparable a la de los modelos de A para esos factores. [PENDIENTE B] más iteraciones del LSTM (L = 48/336, 3/12/30 meses, más épocas) si el tiempo alcanza.

**Curvas de pérdida** (≥ 3 iteraciones): `figs/curvas_perdida_A.png` y `figs/curvas_perdida_recurrentes_B.png`. ¿Una pérdida de entrenamiento más baja implica siempre un mejor MASE en validación? **No.** El MLP con L = 336 sigue bajando la pérdida de entrenamiento (0.220 → 0.199) mientras la de validación se estanca y oscila en ≈0.193–0.196 después de la época 9; DLinear y NLinear convergen en 2–4 épocas y su pérdida queda fija porque el límite es su capacidad, no el sobreajuste.

---

## V. Comparación de estrategias multi-paso

Con los recurrentes de B hay las tres estrategias pedidas (5.1) para el LSTM: **multi-salida** (la RNN lee la ventana y una cabeza emite las 24 horas) y **encoder-decoder** (el decoder recorre las 24 horas futuras recibiendo solo el calendario; no realimenta sus predicciones) y **autorregresiva** (predice una hora y usa esa predicción para la siguiente, sin teacher forcing). En prueba: autorregresiva 0.900 (MAE 38.28 kWh), encoder-decoder 0.958 (40.60) y multi-salida 1.023 (43.09); en validación 0.881, 0.927 y 0.994. Es decir, **la autorregresiva fue la mejor**, y el error no se acumuló con h (ver abajo). Pero la comparación tiene una confusión: la autorregresiva se entrenó con batch 128 contra 1024 de las otras, o sea ≈8 veces más actualizaciones de pesos por época, y cada época le costó 596–645 s contra 31–37 s. No podemos atribuir la ventaja a la estrategia y no al mayor número de pasos de gradiente; para separarlo habría que reentrenar las otras con batch 128 o ésta con 1024. Además, con una sola semilla, la diferencia entre encoder-decoder y multi-salida (≈6 %) y la que hay entre LSTM y GRU (<0.001) no distinguen azar. [PENDIENTE B] la gráfica de MAE por h con las tres curvas.

Lo que ya podemos decir con los modelos de A (todos multi-salida/directos, sin acumulación de errores): el error **no crece monótonamente** con h. Como el origen es siempre las 00:00, h = hora del día + 1, así que el error por paso del horizonte y el error por hora del día son lo mismo: sube hacia el mediodía–tarde (consumo alto y más variable) y baja al final del día. Un modelo autorregresivo mostraría además la acumulación de errores; por eso hay que compararlo contra este perfil.

MASE por paso del horizonte (prueba):

| Modelo | h = 1 | h = 6 | h = 12 | h = 18 | h = 24 |
|---|---|---|---|---|---|
| LightGBM | 0.57 | 0.64 | 0.97 | 1.03 | 0.80 |
| MLP | 0.61 | 0.73 | 0.95 | 1.15 | 0.87 |
| DLinear | 0.58 | 0.73 | 1.06 | 1.13 | 0.85 |
| NLinear | 0.57 | 0.65 | 1.08 | 1.14 | 0.83 |
| MSTL | 0.61 | 0.76 | 1.07 | 1.18 | 0.85 |
| LSTM multi-salida | 0.77 | 0.94 | 1.09 | 1.12 | 1.05 |
| LSTM autorregresivo | 0.68 | 0.69 | 0.99 | 1.04 | 0.90 |
| GRU multi-salida | 0.78 | 0.92 | 1.08 | 1.11 | 1.04 |
| LSTM seq2seq | 0.79 | 0.80 | 1.08 | 1.08 | 0.97 |
| GRU seq2seq | 0.74 | 0.74 | 1.07 | 1.10 | 1.03 |
| Seasonal naive 24 h | 0.86 | 0.75 | 1.31 | 1.33 | 1.02 |

Los recurrentes de salida directa empiezan peor que los modelos de A en h = 1 (0.74–0.79 vs 0.57–0.61) y se parecen a ellos hacia h = 12–18 (1.07–1.12 vs 0.95–1.15). El autorregresivo es el más cercano: 0.68 en h = 1 y 0.99–1.04 en h = 12–18, con el error subiendo hacia la tarde y no creciendo con h.

---

## VI. Tabla comparativa de todos los modelos

Ordenada por MASE en prueba (2014-10-01 a 2014-12-31; 92 orígenes × 24 h × 100 clientes; H = 24). MAE y RMSE en kWh; WAPE = Σ|e|/Σ|y|. MASE escalado con el seasonal naive de 24 h en entrenamiento. *MASE con H = 168 no aplica* (todos los modelos pronostican 24 h).

| Modelo | Familia | L (h) | Multi-paso | Cov. | Params totales / entrenables | t entren. (s) | t infer. 92 orígenes (s) | MAE | RMSE | WAPE | MASE | MASE 22–28 dic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM global | Árboles | rezagos 1–24, 168 | directa (h como variable) | sí | 126 000 hojas | 189 | 3.1 | 33.93 | 164.6 | 6.29 % | **0.850** | 1.54 |
| MLP | MLP | 336 | multi-salida | sí | 533 528 | 177 | 4.5 | 37.50 | 175.7 | 6.95 % | 0.893 | 1.64 |
| LSTM autorregresivo | Recurrente | 168 | autorregresiva (sin teacher forcing) | sí | 21 825 | 9 482‡ | 3.6 | 38.28 | 188.3 | 7.10 % | 0.900 | 1.78 |
| DLinear | Lineal | 336 | multi-salida | no | 16 176 | 26 | 2.3 | 37.58 | 184.7 | 6.97 % | 0.917 | 2.02 |
| NLinear | Lineal | 336 | multi-salida | no | 8 088 | 15 | 2.1 | 37.62 | 184.7 | 6.98 % | 0.928 | 1.99 |
| MSTL (24+168) | Baseline | 1 344* | local | no | – | 1 245† | 1 245† | 38.90 | 193.1 | 7.21 % | 0.952 | 2.18 |
| LSTM seq2seq | Recurrente | 168 | encoder-decoder | sí | 35 905 | 587‡ | 1.9 | 40.60 | 192.3 | 7.53 % | 0.958 | 1.67 |
| GRU seq2seq | Recurrente | 168 | encoder-decoder | sí | 26 945 | 600‡ | 1.9 | 41.26 | 199.5 | 7.65 % | 0.959 | 1.83 |
| GRU multi-salida | Recurrente | 168 | multi-salida | sí | 29 336 | 505‡ | 1.9 | 42.38 | 183.4 | 7.86 % | 1.023 | 1.80 |
| LSTM multi-salida | Recurrente | 168 | multi-salida | sí | 33 624 | 490‡ | 1.9 | 43.09 | 192.3 | 7.99 % | 1.023 | 1.79 |
| Seasonal naive 24 h | Baseline | 24 | local | no | – | – | 0.9 | 45.87 | 236.3 | 8.50 % | 1.088 | 2.47 |
| Seasonal naive 168 h | Baseline | 168 | local | no | – | – | 0.9 | 43.17 | 192.5 | 8.00 % | 1.149 | 1.67 |
| Naive | Baseline | 1 | local | no | – | – | 0.9 | 174.4 | 531.8 | 32.3 % | 4.273 | 4.35 |
| [PENDIENTE B] N-HiTS/N-BEATS, TCN, PatchTST, TFT/iTransformer, DeepAR, Chronos-Bolt | | | | | | | | | | | | |

\* ventana de 8 semanas con la que se reajusta en cada origen. † el ajuste ocurre dentro de la inferencia, en cada uno de los 92 orígenes. Hardware: Apple M5; PyTorch en GPU (MPS), LightGBM y StatsForecast en CPU (los tiempos no son estrictamente comparables entre familias). Chronos: solo inferencia. ‡ Redes recurrentes entrenadas por B en una GPU Tesla T4 (15 épocas; el autorregresivo, con batch 128 y repartido en varias sesiones de Colab); sus tiempos no son comparables con los de A (Apple M5).

**Reentrenamiento mensual** (ventana expansiva, 1-oct, 1-nov, 1-dic): ¿cambia el error? Muy poco. LightGBM 0.8499 → 0.8485 (sin cambio en octubre, como corresponde: mismos datos); MLP 0.8927 → 0.8748, con la mejora concentrada en noviembre (0.750 → 0.720) y diciembre (1.077 → 1.043). El RMSE empeora ligeramente en ambos (MLP 175.7 → 182.9; LightGBM 164.6 → 167.0). El MLP no es reproducible bit a bit en la GPU: con los mismos datos en octubre su MASE varía en ±0.01 (0.846 vs 0.857), así que la mejora del MLP es orientativa con una sola corrida. Los baselines estadísticos ya se reajustan en cada origen, por lo que no se repiten. Con los recurrentes de B no hace falta repetirlo: el mejor modelo de deep learning por MASE de **validación** sigue siendo el MLP (0.795 vs 0.918 del mejor recurrente, GRU seq2seq), que ya está reentrenado. [PENDIENTE B] repetirlo solo si sube un modelo (NeuralForecast) que supere al MLP en validación.

---

## VII. Resultados probabilísticos

[PENDIENTE B] DeepAR (NeuralForecast o GluonTS). Tabla pedida: pinball loss (media sobre cuantiles) y cobertura del intervalo P10–P90 en prueba; la cobertura nominal es 80 %. Para que `05_comparacion.ipynb` los calcule solo, el parquet debe incluir las columnas `q10`, `q50`, `q90` (y opcionalmente más `qNN`) además de `y_hat`.

| Modelo | Pinball loss (kWh) | Cobertura P10–P90 (objetivo 80 %) | MASE (mediana) |
|---|---|---|---|
| DeepAR | [PENDIENTE B] | [PENDIENTE B] | [PENDIENTE B] |
| Chronos-Bolt (cuantiles) | [PENDIENTE B] | [PENDIENTE B] | [PENDIENTE B] |

---

## VIII. Gráficas

1. **MASE de prueba vs tiempo de entrenamiento (log)** — `figs/pareto_mase_tiempo.png`. La frontera de Pareto la forman seasonal naive 24 h, NLinear, DLinear, MLP y LightGBM; MSTL queda fuera (1 245 s por ajustar en cada origen, sin ganar a los lineales). Los cinco recurrentes quedan dominados. Los de salida directa, porque DLinear y NLinear tienen mejor MASE (0.917–0.928 vs 0.958–1.023) con 15–26 s contra 490–600 s. El autorregresivo, porque el MLP lo supera en MASE (0.893 vs 0.900) con 177 s contra ≈9 500 s (hardware distinto, pero el orden de magnitud no cambia). El naive (MASE 4.3) se omite del eje.
2. **MAE por paso del horizonte (h = 1…24)** — `figs/mae_por_horizonte.png` (una curva por modelo).
3. **MASE vs meses de historia** — `figs/mase_vs_historia.png` (seasonal naive, LightGBM, DLinear y, de A, MLP y NLinear). Los recurrentes solo se entrenaron con todos los datos, por lo que no aparecen en esta figura. [PENDIENTE B] PatchTST y Chronos como línea de referencia.

Además, para la sección IV: `figs/curvas_perdida_A.png`. Para ahorrar espacio (límite de 5 páginas), conviene poner las cuatro figuras en una cuadrícula de 2 × 2 de ≈ 6 cm de alto.

---

## IX. Análisis

(Respuestas con los modelos de A; las marcadas [PENDIENTE B] dependen de sus modelos.)

- **¿Qué modelo ganó y por cuánto superó al seasonal naive?** LightGBM global (MASE 0.850): 21.8 % menos MASE y 26 % menos MAE y WAPE que el seasonal naive de 24 h (1.088). Cuesta 189 s de entrenamiento en CPU y 3 s de inferencia sobre los 92 orígenes, frente a 0.9 s del baseline. El mantenimiento es bajo: reentrenar mensualmente casi no mejora (0.8499 → 0.8485), así que bastaría reentrenar con poca frecuencia. Con ~26 % menos error de pronóstico day-ahead, la mejora justifica el costo. Ningún recurrente lo alcanza (mejor: 0.958). [PENDIENTE B] confirmar contra NeuralForecast/Chronos cuando estén.
- **¿DLinear compite con los Transformers?** DLinear (0.917) y NLinear (0.928), con 8–16 mil parámetros y sin covariables, quedan por delante de MSTL y a solo 0.03–0.08 de MASE del MLP y LightGBM. Además superan a los cuatro recurrentes de salida directa (0.958–1.023) con un tiempo de entrenamiento unas 20–40 veces menor (15–26 s vs 490–600 s). Frente al LSTM autorregresivo (0.900), DLinear y NLinear quedan 0.017–0.028 por detrás en MASE, pero ese modelo necesitó ≈9 500 s de entrenamiento. Los tiempos son de hardware distinto, así que son solo un orden de magnitud. [PENDIENTE B] compararlo con PatchTST e iTransformer/TFT y decir en qué horizonte.
- **¿Qué estrategia multi-paso funcionó mejor y cómo crece el error con h?** Con el LSTM (B), gana la autorregresiva (0.900), luego el encoder-decoder (0.958) y por último la multi-salida (1.023), con la confusión de batch (128 vs 1024) descrita en la sección V. En todas el error es más bajo en h ≤ 6 y sube hacia el mediodía–tarde. El error de la autorregresiva **no se acumula** con h: va de 0.68 en h = 1 a 0.99–1.04 en h = 12–18 y baja a 0.90 en h = 24, es decir, sigue el perfil horario y no un crecimiento por paso; no se ve la acumulación de errores que se esperaría de este esquema, aunque con una sola semilla no podemos descartar que se deba a otra causa. En los modelos directos de A el error no se acumula: sube hacia el mediodía–tarde (MASE 0.57–0.64 en h ≤ 6, 0.95–1.08 en h = 12, ≈1.03–1.18 en h = 18) y baja a ≈0.8–0.87 en h = 24; h es la hora del día, así que refleja la dificultad de cada hora, no el horizonte.
- **Efecto de L, normalización e historia.** Con L = 336 el MASE de validación mejora en los tres modelos respecto de L = 48 (MLP 0.868 → 0.795; DLinear 0.912 → 0.824; NLinear 0.912 → 0.822): ver la semana anterior importa. Solo probamos z-score por serie con estadísticos de entrenamiento; no comparamos contra normalización global ni por ventana (RevIN) [PENDIENTE B si lo hace en el LSTM]. Con 3 meses de historia, el MLP (533 mil parámetros, MASE 1.41) y LightGBM (1.00) quedan cerca o por encima del seasonal naive, mientras los lineales casi no cambian (≈0.91): los modelos más flexibles necesitan más datos. Desde 12 meses LightGBM y el MLP (≈0.85) superan a los lineales y más historia no cambia el resultado. Con un solo entrenamiento no afirmamos diferencias menores a ≈0.01.
- **¿Global o uno por cliente?** [PENDIENTE B] (5.3, MASE en 10 clientes).
- **¿Qué tan cerca quedó Chronos zero-shot?** [PENDIENTE B].
- **Navidad y Año Nuevo.** Todos los modelos empeoran en la semana 22–28 dic: el MASE sube a 1.5–2.5 (casi el doble que en el resto de la prueba). Los dos modelos que usan calendario (LightGBM 1.54 y MLP 1.64) lo hacen mejor que los lineales sin calendario (≈2.0) y que MSTL (2.18), pero el seasonal naive de 168 h (1.67) también supera a esos y los modelos difieren en más cosas que las covariables, así que no podemos atribuirlo solo al feriado. Reentrenar hasta el 30-nov no ayuda en esa semana porque aún no ha visto ningún dato de diciembre (LightGBM 1.542 → 1.543; MLP 1.636 → 1.597). El consumo agregado cae 19 % en 24–26 dic y 9 % el 31 dic, y el 25-dic es 54 % menor que un día laboral típico: un cambio de régimen que afecta a todos los modelos.
- **Sensibilidad a la semilla.** [PENDIENTE B] (3 semillas del mejor modelo). Evidencia parcial: el MLP de A varió ≈ ±0.01 de MASE entre dos entrenamientos con los mismos datos y semilla.

### Discusión integrada

El resultado dominante es que **la información útil estaba en pocos rezagos y en el calendario, no en la arquitectura**. El modelo global de árboles, que recibe las últimas 24 h, el valor de hace 24 y 168 h del instante objetivo y el calendario, es el mejor (MASE 0.850) y el más barato por punto de MASE ganado. Le sigue el MLP, que usa la ventana completa de 336 h más calendario (0.893), y después dos capas lineales sin calendario (0.917–0.928): pasar de 8 mil a 533 mil parámetros solo compra ≈0.03 de MASE. Los recurrentes de salida directa, con una estructura pensada para secuencias, no mejoran esto (0.958–1.023); el LSTM autorregresivo se acerca (0.900, entre el MLP y DLinear), pero con un costo de ≈9 500 s de entrenamiento. Una lectura posible es que, con una serie diaria muy regular, la tarea consiste en leer los valores de hace 24 y 168 h, algo que una capa lineal o un árbol hacen directamente, y que la recurrencia solo añade costo de entrenamiento (490–9 500 s contra 15–26 s de los lineales). Esto es una hipótesis: B no varió L ni dejó converger las redes, y la autorregresiva usó otro tamaño de batch, así que no podemos separar "la recurrencia no ayuda" de "estaban a medio entrenar" ni "la estrategia autorregresiva es mejor" de "hubo más pasos de gradiente".

Dos patrones se repiten en todos los modelos y dependen de los datos, no del modelo: el error sube hacia el mediodía–tarde (más consumo y más variabilidad) y la semana del 22–28 de diciembre rompe el régimen (MASE 1.5–2.5), con una caída agregada del 19 % en 24–26 dic. Los modelos con calendario (LightGBM, MLP, seq2seq, autorregresivo) lo manejan algo mejor (1.54–1.83) que los lineales sin calendario y MSTL (≈2.0–2.2), pero el seasonal naive de 168 h (1.67) queda en el mismo rango, así que el calendario ayuda sin resolver el problema.

Sobre costo y mantenimiento: reentrenar cada mes cambia muy poco el error (LightGBM −0.0014, MLP −0.018, que es poco más que el ruido de ±0.01 del MLP en GPU), mientras que la **cantidad de historia** sí importa para los modelos flexibles (MLP 1.41 con 3 meses vs 0.85 con 12). Para un operador, la recomendación que sostienen los datos es: LightGBM con ≥12 meses de historia, reentrenado pocas veces al año, y un baseline estacional o un modelo lineal como respaldo barato. Lo que no podemos afirmar todavía, por falta de los modelos de B: el desempeño de Transformers y modelos preentrenados (Chronos), la calidad de los intervalos probabilísticos, el valor de un modelo por cliente y la variación entre semillas.

---

## X. Conclusiones

- Con un protocolo único (backtest con origen móvil diario, 92 orígenes, mismas métricas y mismo panel de 100 clientes), un modelo global de árboles con rezagos y calendario (LightGBM) fue el mejor de los modelos disponibles hasta ahora, con 21.8 % menos MASE que el seasonal naive de 24 h.
- Los modelos lineales simples (DLinear, NLinear) son una alternativa barata y robusta: 15–26 s de entrenamiento, 8–16 mil parámetros, MASE ≈ 0.92–0.93 y poca sensibilidad a la cantidad de historia.
- La ventana de entrada (336 h) y tener al menos ~12 meses de historia importan más que reentrenar mensualmente.
- La semana de Navidad rompe el régimen para todos los modelos; ningún protocolo de reentrenamiento hasta fin de noviembre lo resuelve.
- Auditoría de fugas con `assert` en el código y escalado solo con estadísticos de entrenamiento evitan sobreestimar el desempeño.
- Redes recurrentes: LSTM y GRU rinden igual (diferencia < 0.001 de MASE). De las tres estrategias del LSTM, la autorregresiva fue la mejor (0.900), luego el encoder-decoder (0.958) y la multi-salida (1.023); la autorregresiva no mostró acumulación de errores con h, pero se entrenó con otro batch (128 vs 1024) y le costó ≈9 500 s, así que la ventaja no puede atribuirse solo a la estrategia. Ningún recurrente alcanza a LightGBM (0.850) ni al MLP (0.893). Los cinco quedaron sin converger (mejor época = 15 de 15), por lo que podrían mejorar con más épocas; no lo afirmamos.
- [PENDIENTE B] Conclusiones sobre Transformers, estrategia autorregresiva, probabilísticos, local vs global, Chronos y semillas.

---

## XI. Anexos

- **A.** Repositorio y archivos: `01_datos.ipynb` (preparación, EDA, ventanas, fugas), `02_backtest.ipynb` (protocolo y métricas), `03_modelos_A.ipynb`, `04_reentrenamiento_A.ipynb`, `05_comparacion.ipynb`; código en `common.py`, `datos.py`, `modelos_clasicos.py`, `modelos_torch.py`, `comparar.py`; scripts `run_baselines.py`, `run_torch.py`, `run_reentreno.py`. De B: `06_recurrentes_B.ipynb`, `modelos_recurrentes.py`, `run_recurrentes.py`. Pronósticos de prueba de cada modelo: `results/preds/<modelo>.parquet` (`unique_id, ds, modelo, y_hat`). Enlace al repositorio: [PENDIENTE: agregar al final del PDF].
- **B.** Logs de entrenamiento por época: `results/iteraciones_A.csv` (A) y `results/iteraciones_B.csv` (recurrentes, con su resumen en `iteraciones_B_resumen.csv`); resumen de cada modelo (parámetros, tiempos, hardware): `results/resumen.csv`; tabla comparativa: `results/tabla_comparativa.csv`.
- **C.** Hiperparámetros: MLP 2 capas ocultas de 512, dropout 0.1, entrada = ventana L + calendario futuro; DLinear con media móvil de 25 h; NLinear resta el último valor; Adam lr 1e-3, batch 1024, pérdida L1, semilla 42; LightGBM lr 0.05, 63 hojas, objetivo L1, 2 000 árboles, feature_fraction 0.8, bagging 0.8. Recurrentes (B): 1 capa, estado oculto 64, dropout 0.1, L = 168, Adam lr 1e-3, pérdida L1, 15 épocas, mismo calendario futuro; el seq2seq usa el calendario como entrada del decoder.
- **D.** MAPE: se calcula solo sobre horas con consumo > 0 (3 018 horas, 1.37 % de la prueba, tienen consumo exactamente 0); con consumo cercano a 0 explota (medido con el seasonal naive de 168 h: en la franja 0.1–1 kWh el error relativo medio es 25.8, es decir 2 580 %, y la mediana 1.0), por eso se reporta WAPE como métrica principal.
- **E.** Supuestos y limitaciones: una sola semilla, recurrentes de B sin converger, con una sola configuración de L y con el autorregresivo entrenado con otro batch, StatsForecast reajustado por origen con 8 semanas, LightGBM llegó al tope de 2 000 árboles, y los tiempos de PyTorch (GPU) y de LightGBM/StatsForecast (CPU) no son estrictamente comparables.
