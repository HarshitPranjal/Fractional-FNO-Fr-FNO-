#!/usr/bin/env python3
"""
Priority 4: n_train=900 Extension (All training indices 0..899)
- Cells: S2 alpha=0.8 and S1 alpha=1.2
- Arms: A0, A3, A6, A10
- Seeds: 42, 123, 999, 2024, 2025 (5 seeds)
- Tuned configs: lr=3.0e-3, wd=0.0
- Epochs: 100, batch_size: 125, AdamW, eta_min=1e-5, MPS
- Append to results/review_records.jsonl with stage="final_p4"
- Output tables/tab_p4_n900.tex and results/priority4_n900_stats.csv
"""

import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path and set as current working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import json
import time
import numpy as np
import pandas as pd
import scipy.stats as st
from fr_fno_master import train_one_run

RECORDS_FILE = "results/review_records.jsonl"
os.makedirs("results", exist_ok=True)
os.makedirs("tables", exist_ok=True)

def append_review_record(rec, stage="final_p4"):
    formatted = {
        "system": rec["system"],
        "alpha_true": float(rec["alpha_true"]),
        "arm": rec["arm"],
        "seed": int(rec["seed"]),
        "n_train": 900,
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
        "stage": stage,
    }
    with open(RECORDS_FILE, "a") as f:
        f.write(json.dumps(formatted) + "\n")
    return formatted

cells = [
    ("S2_allen_cahn", 0.8),
    ("S1_burgers", 1.2),
]
arms = ["A0", "A3", "A6", "A10"]
seeds = [42, 123, 999, 2024, 2025]

print("=" * 80)
print("PRIORITY 4: n_train=900 BENCHMARK (40 RUNS, 5 SEEDS)")
print("=" * 80)

p4_records = []
diverged = []
t_start = time.time()

for sys_name, a_val in cells:
    for arm in arms:
        b_lr = 3.0e-3
        b_wd = 0.0
        for s in seeds:
            print(f"[*] n_tr=900: {sys_name} a={a_val} {arm} s={s}...", end=" ", flush=True)
            r_start = time.time()
            rec = train_one_run(sys_name, a_val, arm, 900, s, epochs=100, batch_size=125, base_lr=b_lr, weight_decay=b_wd)
            formatted = append_review_record(rec, stage="final_p4")
            p4_records.append(formatted)
            r_sec = time.time() - r_start
            rel_l2 = formatted["test_rel_l2"]
            if np.isnan(rel_l2) or np.isinf(rel_l2):
                diverged.append((sys_name, a_val, arm, s))
                print(f"DIVERGED in {r_sec:.1f}s")
            else:
                print(f"Done in {r_sec:.1f}s (test_rel_l2={rel_l2:.4f})")

total_p4_sec = time.time() - t_start
print(f"\n[+] Finished 40 runs in {total_p4_sec:.1f}s ({total_p4_sec/60:.2f} min). Diverged: {len(diverged)}")

df_p4 = pd.DataFrame(p4_records)
df_p4.to_csv("results/priority4_n900_raw.csv", index=False)

# Statistics & Paired Differences vs A0
t_crit_4 = st.t.ppf(0.975, df=4)  # 2.7764 for 5 seeds
stat_rows = []

for (sys_name, a_val), grp in df_p4.groupby(["system", "alpha_true"]):
    pivot = grp.pivot(index="seed", columns="arm", values="test_rel_l2")
    a0_vals = pivot["A0"]
    for arm in arms:
        vals = pivot[arm]
        mean = vals.mean()
        std = vals.std(ddof=1)
        ci_half = t_crit_4 * (std / np.sqrt(5))
        
        if arm == "A0":
            mean_diff = 0.0
            std_diff = 0.0
            ci_diff_half = 0.0
            ci_diff_low = 0.0
            ci_diff_high = 0.0
            verdict = "Baseline"
        else:
            diffs = vals - a0_vals
            mean_diff = diffs.mean()
            std_diff = diffs.std(ddof=1)
            ci_diff_half = t_crit_4 * (std_diff / np.sqrt(5))
            ci_diff_low = mean_diff - ci_diff_half
            ci_diff_high = mean_diff + ci_diff_half
            if ci_diff_high < 0:
                verdict = "Better than A0 (p < 0.05)"
            elif ci_diff_low > 0:
                verdict = "Worse than A0 (p < 0.05)"
            else:
                verdict = "Tie / Parity"
                
        stat_rows.append({
            "system": sys_name,
            "alpha_true": a_val,
            "arm": arm,
            "n_train": 900,
            "n_seeds": 5,
            "test_rel_l2_mean": mean,
            "test_rel_l2_std_ddof1": std,
            "ci95_half_width": ci_half,
            "ci95_low": mean - ci_half,
            "ci95_high": mean + ci_half,
            "mean_diff_vs_A0": mean_diff,
            "ci95_diff_half": ci_diff_half,
            "ci95_diff_low": ci_diff_low,
            "ci95_diff_high": ci_diff_high,
            "verdict": verdict,
        })

df_p4_stats = pd.DataFrame(stat_rows)
df_p4_stats.to_csv("results/priority4_n900_stats.csv", index=False)
print("\n" + "=" * 80)
print("PRIORITY 4 RESULTS: n_train=900 STATS & PAIRED DIFFERENCES (5 SEEDS)")
print("=" * 80)
print(df_p4_stats.to_string())

tex_p4 = r"""\begin{table*}[t]
\centering
\small
\caption{Full-Data Evaluation ($N_{\text{tr}}=900$, 100 Epochs) on S2 Allen-Cahn ($\alpha=0.8$) and S1 Burgers ($\alpha=1.2$) across 5 Seeds.}
\label{tab:p4_n900}
\begin{tabular}{llcccc}
\toprule
System & Arm & Test Rel. $L^2$ (Mean $\pm$ Std) & 95\% CI & Paired Diff vs. $A_0$ [95\% CI] & Verdict \\
\midrule
"""
for _, r in df_p4_stats.iterrows():
    sys_str = "S2 (AC $\\alpha=0.8$)" if "S2" in r["system"] else "S1 (Burgers $\\alpha=1.2$)"
    diff_str = "--" if r["arm"] == "A0" else f"{r['mean_diff_vs_A0']:+.4f} [{r['ci95_diff_low']:+.4f}, {r['ci95_diff_high']:+.4f}]"
    tex_p4 += f"{sys_str} & {r['arm']} & {r['test_rel_l2_mean']:.4f} $\\pm$ {r['test_rel_l2_std_ddof1']:.4f} & [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] & {diff_str} & {r['verdict']} \\\\\n"
tex_p4 += r"""\bottomrule
\end{tabular}
\end{table*}
"""
with open("tables/tab_p4_n900.tex", "w") as f:
    f.write(tex_p4)
print("\n[+] Wrote tables/tab_p4_n900.tex")
