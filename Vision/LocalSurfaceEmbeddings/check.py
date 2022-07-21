from pathlib import Path
import numpy as np


# l = ['06', '07', '08', '09', '23', '27', '29']

model_dir = Path.home() / "data" / "t-less_v2" / "lse"

for i in model_dir.glob("*.npy"):
    if i.stem[-2:] in l:
        v = np.load(i)
        for j in range(v.shape[0]):
            if np.abs(v[j][4:].mean()) > 1e-8:
                print("ERROR", i, v[j][4:].mean())

data_dir = Path.home() / "data" / "t-less_v2" / "train_primesense"

for obj_dir in data_dir.iterdir():
    for lse_file in (obj_dir / "lse_embeddings").iterdir():
        lse = np.load(lse_file)
        if np.any(np.isnan(lse)):
            print("ERROR Type 1", lse_file)
        elif np.abs(lse.mean()) > 1e-8:
            print("ERROR Type 2", lse.mean())
