"""Resume results/semillas.csv en informe/semillas.json (media y desv. est. muestral por modelo) para el informe."""
import json
import pandas as pd

d = pd.read_csv("results/semillas.csv")
d = d[d["MASE"] < 3]                      # descarta corridas corruptas (GPU del Mac sin memoria)
out = {}
for m, g in d.groupby("modelo"):
    if len(g) < 3:
        continue
    g = g.sort_values("semilla")
    out[m] = {"vals": g["MASE"].tolist(), "mean": g["MASE"].mean(), "std": g["MASE"].std(),
              "mae_mean": g["MAE"].mean(), "mae_std": g["MAE"].std()}
json.dump(out, open("informe/semillas.json", "w"), indent=1)
print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in out.items()})
