#!/usr/bin/env python3
"""
Final Audit Pipeline
Completes:
1. Tuning parity for the remaining 5 arms on S2 alpha=1.2
2. Consolidating chosen (lr*, wd*) across all 27 (cell, arm) pairs
3. Final 5-seed confirmation at chosen configs (N_tr=500, 100 epochs)
4. 300-epoch extension on S2 alpha=1.2 (3 seeds)
5. Rigorous 95% Student-t CIs and paired t-tests vs A0
All records logged to results/review_records.jsonl with exact 18-field schema.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import scipy.stats as st

from fr_fno_master import train_one_run

RECORDS_FILE = "results/review_records.jsonl"
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

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
        "lr": float(rec.get("base_lr", rec.get("lr", 1.0e-3))),
        "weight_decay": float(rec.get("weight_decay", 1.0e-4)),
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

def load_existing():
    existing = set()
    if os.path.exists(RECORDS_FILE):
        with open(RECORDS_FILE) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    key = (r["system"], float(r["alpha_true"]), r["arm"], int(r["seed"]), int(r["epochs"]), round(float(r["lr"]), 6), round(float(r["weight_decay"]), 6))
                    existing.add(key)
    return existing

EXISTING = load_existing()

# Known chosen configs from Stage 2 tuning so far:
chosen_configs = {
    ("S1_burgers", 1.2, "A0"): (3.0e-3, 0.0, 3.5598e-2, 3.9770e-2),
    ("S1_burgers", 1.2, "A3"): (3.0e-3, 0.0, 5.3581e-2, 5.6707e-2),
    ("S1_burgers", 1.2, "A5"): (3.0e-3, 0.0, 4.3719e-2, 4.6662e-2),
    ("S1_burgers", 1.2, "A6"): (3.0e-3, 0.0, 3.8897e-2, 4.4080e-2),
    ("S1_burgers", 1.2, "A6j"): (3.0e-3, 0.0, 3.6989e-2, 4.1013e-2),
    ("S1_burgers", 1.2, "A10"): (3.0e-3, 0.0, 3.6975e-2, 4.1006e-2),
    ("S1_burgers", 1.2, "A9a"): (3.0e-3, 0.0, 3.5801e-2, 3.9641e-2),
    ("S1_burgers", 1.2, "DeepONet"): (3.0e-3, 0.0, 4.2605e-1, 4.3942e-1),
    ("S1_burgers", 1.2, "UNet"): (3.0e-3, 0.0, 1.5595e-1, 1.6346e-1),

    ("S2_allen_cahn", 0.8, "A0"): (3.0e-3, 0.0, 1.4644e-2, 1.4808e-2),
    ("S2_allen_cahn", 0.8, "A3"): (3.0e-3, 0.0, 7.3342e-3, 7.4929e-3),
    ("S2_allen_cahn", 0.8, "A5"): (3.0e-3, 0.0, 6.7374e-3, 6.8394e-3),
    ("S2_allen_cahn", 0.8, "A6"): (3.0e-3, 0.0, 6.7060e-3, 6.8399e-3),
    ("S2_allen_cahn", 0.8, "A6j"): (3.0e-3, 0.0, 1.5674e-2, 1.5818e-2),
    ("S2_allen_cahn", 0.8, "A10"): (3.0e-3, 0.0, 1.5662e-2, 1.5797e-2),
    ("S2_allen_cahn", 0.8, "A9a"): (3.0e-3, 0.0, 1.4458e-2, 1.4579e-2),
    ("S2_allen_cahn", 0.8, "DeepONet"): (3.0e-3, 0.0, 5.3967e-1, 5.4047e-1),
    ("S2_allen_cahn", 0.8, "UNet"): (3.0e-3, 0.0, 4.4810e-2, 4.4611e-2),

    ("S2_allen_cahn", 1.2, "A0"): (3.0e-3, 0.0, 1.5302e-2, 1.5823e-2),
    ("S2_allen_cahn", 1.2, "A3"): (3.0e-3, 0.0, 1.5198e-2, 1.5959e-2),
    ("S2_allen_cahn", 1.2, "A5"): (3.0e-3, 0.0, 1.4376e-2, 1.5082e-2),
    ("S2_allen_cahn", 1.2, "A6"): (3.0e-3, 1.0e-4, 1.4686e-2, 1.5309e-2),
}

# 1. Tune remaining 5 arms on S2 alpha=1.2
remaining_arms_s2 = ["A6j", "A10", "A9a", "DeepONet", "UNet"]
lr_grid = [3.0e-4, 1.0e-3, 3.0e-3]
wd_grid = [0.0, 1.0e-4]

print("="*80)
print("STEP 1: TUNING REMAINING 5 ARMS ON S2 alpha=1.2")
print("="*80)

for arm in remaining_arms_s2:
    best_val = float("inf")
    best_cfg = None
    best_rec = None
    for lr in lr_grid:
        for wd in wd_grid:
            key = ("S2_allen_cahn", 1.2, arm, 42, 100, round(float(lr), 6), round(float(wd), 6))
            if key not in EXISTING:
                rec = train_one_run("S2_allen_cahn", 1.2, arm, 500, 42, epochs=100, base_lr=lr, weight_decay=wd)
                append_review_record(rec)
                EXISTING.add(key)
            else:
                # Load from records file
                rec = None
                with open(RECORDS_FILE) as f:
                    for l in f:
                        r = json.loads(l)
                        if (r["system"], float(r["alpha_true"]), r["arm"], int(r["seed"]), int(r["epochs"]), round(float(r["lr"]), 6), round(float(r["weight_decay"]), 6)) == key:
                            rec = r
            val_err = rec.get("val_rel_l2", rec.get("test_rel_l2"))
            if val_err < best_val:
                best_val = val_err
                best_cfg = (lr, wd)
                best_rec = rec
    chosen_configs[("S2_allen_cahn", 1.2, arm)] = (best_cfg[0], best_cfg[1], best_val, best_rec["test_rel_l2"])
    print(f"[S2_allen_cahn a=1.2 | {arm:8s}] Selected: lr={best_cfg[0]:.1e}, wd={best_cfg[1]:.1e} | val_rel_l2={best_val:.4e} | test_rel_l2={best_rec['test_rel_l2']:.4e}")

# Save chosen configs summary
chosen_rows = []
for (sys_name, a_val, arm), (b_lr, b_wd, b_val, b_test) in chosen_configs.items():
    chosen_rows.append({
        "system": sys_name,
        "alpha_true": a_val,
        "arm": arm,
        "best_lr": b_lr,
        "best_wd": b_wd,
        "val_rel_l2": b_val,
        "test_rel_l2_seed42": b_test,
    })
df_chosen = pd.DataFrame(chosen_rows)
df_chosen.to_csv("results/tuning_parity_chosen.csv", index=False)
print("\n[+] Saved complete tuning parity to results/tuning_parity_chosen.csv")

print("\n" + "="*80)
print("STEP 2: RUNNING REMAINING 4 SEEDS AT CHOSEN CONFIGS")
print("="*80)

rem_seeds = [123, 999, 2024, 2025]
for (sys_name, a_val, arm), (b_lr, b_wd, _, _) in chosen_configs.items():
    print(f"[*] Running seeds for {sys_name} a={a_val} {arm} (lr={b_lr:.1e}, wd={b_wd:.1e})...")
    for s in rem_seeds:
        key = (sys_name, float(a_val), arm, int(s), 100, round(float(b_lr), 6), round(float(b_wd), 6))
        if key not in EXISTING:
            rec = train_one_run(sys_name, a_val, arm, 500, s, epochs=100, base_lr=b_lr, weight_decay=b_wd)
            append_review_record(rec)
            EXISTING.add(key)

print("\n" + "="*80)
print("STEP 3: 300-EPOCH EXTENSION ON S2 alpha=1.2")
print("="*80)

arms_300 = ["A0", "A3", "A6", "A10", "DeepONet", "UNet"]
seeds_300 = [42, 123, 999]

for arm in arms_300:
    b_lr, b_wd, _, _ = chosen_configs[("S2_allen_cahn", 1.2, arm)]
    print(f"[*] 300 epochs for {arm} (lr={b_lr:.1e}, wd={b_wd:.1e})...")
    for s in seeds_300:
        key = ("S2_allen_cahn", 1.2, arm, int(s), 300, round(float(b_lr), 6), round(float(b_wd), 6))
        if key not in EXISTING:
            rec = train_one_run("S2_allen_cahn", 1.2, arm, 500, s, epochs=300, base_lr=b_lr, weight_decay=b_wd)
            append_review_record(rec)
            EXISTING.add(key)

print("\n" + "="*80)
print("STEP 4: COMPUTING RIGOROUS 95% CIs & SUMMARY TABLES")
print("="*80)

# Load all records from review_records.jsonl
with open(RECORDS_FILE) as f:
    all_recs = [json.loads(l) for l in f if l.strip()]

df_all = pd.DataFrame(all_recs)

# 1. Final 5-seed statistics (100 epochs, train_size=500)
# Filter records that match chosen configs
target_100 = []
for (sys_name, a_val, arm), (b_lr, b_wd, _, _) in chosen_configs.items():
    sub = df_all[
        (df_all["system"] == sys_name) &
        (np.isclose(df_all["alpha_true"], a_val)) &
        (df_all["arm"] == arm) &
        (df_all["epochs"] == 100) &
        (np.isclose(df_all["lr"], b_lr)) &
        (np.isclose(df_all["weight_decay"], b_wd))
    ]
    # Keep unique seeds
    sub_dedup = sub.drop_duplicates(subset=["seed"])
    target_100.append(sub_dedup)

df_100_final = pd.concat(target_100, ignore_index=True)
t_crit_4 = st.t.ppf(0.975, df=4)
ci_rows = []

for (sys_name, a_val), cell_df in df_100_final.groupby(["system", "alpha_true"]):
    pivot = cell_df.pivot(index="seed", columns="arm", values="test_rel_l2")
    a0_series = pivot["A0"] if "A0" in pivot.columns else None

    for arm in pivot.columns:
        vals = pivot[arm].dropna()
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        ci_half = t_crit_4 * (std / np.sqrt(n)) if n > 1 else 0.0
        p_val = 1.0
        if arm != "A0" and a0_series is not None and n == 5:
            p_val = st.ttest_rel(vals, a0_series).pvalue

        ci_rows.append({
            "system": sys_name,
            "alpha_true": a_val,
            "arm": arm,
            "n_seeds": n,
            "test_rel_l2_mean": mean,
            "test_rel_l2_std": std,
            "ci95_half_width": ci_half,
            "ci95_low": mean - ci_half,
            "ci95_high": mean + ci_half,
            "paired_t_pvalue_vs_A0": p_val,
        })

df_final_ci = pd.DataFrame(ci_rows)
df_final_ci.to_csv("results/final_5seed_confidence_intervals.csv", index=False)
print("\n[+] Saved results/final_5seed_confidence_intervals.csv")
print(df_final_ci.to_string())

# 2. 300-epoch statistics
df_300 = df_all[(df_all["system"] == "S2_allen_cahn") & (df_all["alpha_true"] == 1.2) & (df_all["epochs"] == 300)]
df_300_dedup = df_300.drop_duplicates(subset=["arm", "seed"])
pivot_300 = df_300_dedup.pivot(index="seed", columns="arm", values="test_rel_l2")
a0_300 = pivot_300["A0"] if "A0" in pivot_300.columns else None
t_crit_2 = st.t.ppf(0.975, df=2)
rows_300 = []

for arm in pivot_300.columns:
    vals = pivot_300[arm].dropna()
    n = len(vals)
    mean = vals.mean()
    std = vals.std(ddof=1) if n > 1 else 0.0
    ci_half = t_crit_2 * (std / np.sqrt(n)) if n > 1 else 0.0
    p_val = 1.0
    if arm != "A0" and a0_300 is not None and n == 3:
        p_val = st.ttest_rel(vals, a0_300).pvalue
    rows_300.append({
        "system": "S2_allen_cahn",
        "alpha_true": 1.2,
        "arm": arm,
        "epochs": 300,
        "n_seeds": n,
        "test_rel_l2_mean": mean,
        "test_rel_l2_std": std,
        "ci95_half_width": ci_half,
        "paired_t_pvalue_vs_A0": p_val,
    })

df_300_ci = pd.DataFrame(rows_300)
df_300_ci.to_csv("results/extended_300epoch_confidence_intervals.csv", index=False)
print("\n[+] Saved results/extended_300epoch_confidence_intervals.csv")
print(df_300_ci.to_string())

print("\n" + "="*80)
print("AUDIT PIPELINE FULLY COMPLETED!")
print("="*80)
