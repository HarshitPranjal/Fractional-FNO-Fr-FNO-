#!/usr/bin/env python3
"""
Tier C1 Extension:
S1 Burgers and S2 Allen-Cahn at alpha=1.6 and alpha=1.9
Arms: A0, A3, A6, A10
Seeds: 42, 123, 999
N_tr = 500, 100 epochs, AdamW, batch_size = 125
"""

import os
import json
import pandas as pd
import scipy.stats as st
from fr_fno_master import train_one_run

RECORDS_FILE = "results/review_records.jsonl"

def append_review_record(rec):
    formatted = {
        "system": rec["system"],
        "alpha_true": float(rec["alpha_true"]),
        "arm": rec["arm"],
        "seed": int(rec["seed"]),
        "n_train": int(rec.get("n_train", rec.get("train_size", 500))),
        "epochs": int(rec["epochs"]),
        "batch_size": int(rec.get("batch_size", 125)),
        "optimizer": "AdamW",
        "lr": float(rec.get("base_lr", rec.get("lr", 3.0e-3))),
        "weight_decay": float(rec.get("weight_decay", 0.0)),
        "test_rel_l2": float(rec["test_rel_l2"]),
        "test_mse": float(rec["test_mse"]),
        "test_smape": float(rec["test_smape"]),
        "test_mape_eps": float(rec["test_mape_eps"]),
        "test_mape_clamped": float(rec["test_mape_clamped"]),
        "params": int(rec.get("trainable_params", rec.get("params", 287361))),
        "train_seconds": float(rec.get("wall_clock_sec", rec.get("train_seconds", 0.0))),
        "infer_ms_per_sample": float(rec["infer_ms_per_sample"]),
    }
    with open(RECORDS_FILE, "a") as f:
        f.write(json.dumps(formatted) + "\n")
    return formatted

cells = [
    ("S1_burgers", 1.6),
    ("S1_burgers", 1.9),
    ("S2_allen_cahn", 1.6),
    ("S2_allen_cahn", 1.9),
]

arms = ["A0", "A3", "A6", "A10"]
seeds = [42, 123, 999]

print("Starting Tier C1 runs...")
c1_records = []

for sys_name, a_val in cells:
    for arm in arms:
        wd = 1.0e-4 if (sys_name == "S2_allen_cahn" and arm == "A6") else 0.0
        for s in seeds:
            print(f"[*] C1: {sys_name} alpha={a_val} {arm} seed={s}...")
            rec = train_one_run(sys_name, a_val, arm, 500, s, epochs=100, base_lr=3.0e-3, weight_decay=wd)
            formatted = append_review_record(rec)
            c1_records.append(formatted)

df_c1 = pd.DataFrame(c1_records)
df_c1.to_csv("results/tier_c1_summary.csv", index=False)
print("Saved results/tier_c1_summary.csv")

# Print summary
summary = df_c1.groupby(["system", "alpha_true", "arm"])["test_rel_l2"].agg(["count", "mean", "std"])
print(summary.to_string())
print("Tier C1 complete!")
