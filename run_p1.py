#!/usr/bin/env python3
"""
Priority 1 Execution:
1a: 300 epochs on S2 alpha=0.8 and S1 alpha=1.2 for arms A0, A3, A5, A6, A10 across seeds 42, 123, 999.
1b: Selected hyperparameters LaTeX table (tables/tab_selected_hparams.tex) and plain text.
All runs appended to results/review_records.jsonl with stage="final_p1".
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
os.makedirs("tables", exist_ok=True)

def append_review_record(rec, stage="final_p1"):
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
        "stage": stage,
    }
    with open(RECORDS_FILE, "a") as f:
        f.write(json.dumps(formatted) + "\n")
    return formatted

# -----------------------------------------------------------------------------
# 1b: Output Selected Hyperparameters Table (tables/tab_selected_hparams.tex)
# -----------------------------------------------------------------------------
chosen_df = pd.read_csv("results/tuning_parity_chosen.csv")

tex_content = r"""\begin{table}[t]
\centering
\small
\caption{Validation-Selected Hyperparameters and Errors ($N_{\text{tr}}=500$, 100 Epochs, Seed 42).}
\label{tab:selected_hparams}
\begin{tabular}{llcccc}
\toprule
System & Arm & Selected LR & Selected WD & Val Rel. $L^2$ & Test Rel. $L^2$ \\
\midrule
"""
for _, r in chosen_df.iterrows():
    sys_clean = "S1 (Burgers)" if "S1" in r["system"] else f"S2 (AC $\\alpha={r['alpha_true']}$)"
    tex_content += f"{sys_clean} & {r['arm']} & {r['best_lr']:.1e} & {r['best_wd']:.1e} & {r['val_rel_l2']:.4f} & {r['test_rel_l2_seed42']:.4f} \\\\\n"
tex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open("tables/tab_selected_hparams.tex", "w") as f:
    f.write(tex_content)
print("[+] Wrote tables/tab_selected_hparams.tex")
print("\nSelected Hyperparameters (Plain Text):")
print(chosen_df.to_string())

# -----------------------------------------------------------------------------
# 1a: Run 300 epochs on S2 alpha=0.8 and S1 alpha=1.2
# -----------------------------------------------------------------------------
cells = [
    ("S2_allen_cahn", 0.8),
    ("S1_burgers", 1.2),
]
arms = ["A0", "A3", "A5", "A6", "A10"]
seeds = [42, 123, 999]

print("\n" + "="*80)
print("PRIORITY 1a: 300 EPOCHS EXTENSION (30 RUNS)")
print("="*80)

p1_records = []
diverged = []
t_start = time.time()

for sys_name, a_val in cells:
    for arm in arms:
        # Retrieve chosen lr and wd
        sub = chosen_df[(chosen_df["system"] == sys_name) & (np.isclose(chosen_df["alpha_true"], a_val)) & (chosen_df["arm"] == arm)]
        b_lr = float(sub["best_lr"].values[0])
        b_wd = float(sub["best_wd"].values[0])
        for s in seeds:
            print(f"[*] 300ep: {sys_name} a={a_val} {arm} s={s} (lr={b_lr:.1e}, wd={b_wd:.1e})...", end=" ", flush=True)
            r_start = time.time()
            rec = train_one_run(sys_name, a_val, arm, 500, s, epochs=300, base_lr=b_lr, weight_decay=b_wd)
            formatted = append_review_record(rec, stage="final_p1")
            p1_records.append(formatted)
            r_sec = time.time() - r_start
            rel_l2 = formatted["test_rel_l2"]
            if np.isnan(rel_l2) or np.isinf(rel_l2):
                diverged.append((sys_name, a_val, arm, s))
                print(f"DIVERGED in {r_sec:.1f}s")
            else:
                print(f"Done in {r_sec:.1f}s (test_rel_l2={rel_l2:.4f})")

total_p1_sec = time.time() - t_start
print(f"\n[+] Finished 30 runs in {total_p1_sec:.1f}s ({total_p1_sec/60:.2f} min). Diverged runs: {len(diverged)}")

# Save P1 summary
df_p1 = pd.DataFrame(p1_records)
df_p1.to_csv("results/priority1_300epoch_raw.csv", index=False)

# Statistical Summary per cell & arm
t_crit_2 = st.t.ppf(0.975, df=2)
stat_rows = []

for (sys_name, a_val), grp in df_p1.groupby(["system", "alpha_true"]):
    pivot = grp.pivot(index="seed", columns="arm", values="test_rel_l2")
    a0_vals = pivot["A0"]
    for arm in arms:
        vals = pivot[arm]
        mean = vals.mean()
        std = vals.std(ddof=1)
        ci_half = t_crit_2 * (std / np.sqrt(3))
        
        # Paired difference vs A0
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
            ci_diff_half = t_crit_2 * (std_diff / np.sqrt(3))
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
            "epochs": 300,
            "n_seeds": 3,
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

df_p1_stats = pd.DataFrame(stat_rows)
df_p1_stats.to_csv("results/priority1_300epoch_stats.csv", index=False)
print("\n" + "="*80)
print("PRIORITY 1a RESULTS: 300 EPOCHS STATS & PAIRED DIFFERENCES")
print("="*80)
print(df_p1_stats.to_string())

# Also format as LaTeX table
tex_p1 = r"""\begin{table*}[t]
\centering
\small
\caption{Extended 300-Epoch Evaluation on S2 Allen-Cahn ($\alpha=0.8$) and S1 Burgers ($\alpha=1.2$) across 3 Seeds.}
\label{tab:p1_300epoch}
\begin{tabular}{llcccc}
\toprule
System & Arm & Test Rel. $L^2$ (Mean $\pm$ Std) & 95\% CI & Paired Diff vs. $A_0$ [95\% CI] & Verdict \\
\midrule
"""
for _, r in df_p1_stats.iterrows():
    sys_str = "S2 (AC $\\alpha=0.8$)" if "S2" in r["system"] else "S1 (Burgers $\\alpha=1.2$)"
    diff_str = "--" if r["arm"] == "A0" else f"{r['mean_diff_vs_A0']:+.4f} [{r['ci95_diff_low']:+.4f}, {r['ci95_diff_high']:+.4f}]"
    tex_p1 += f"{sys_str} & {r['arm']} & {r['test_rel_l2_mean']:.4f} $\\pm$ {r['test_rel_l2_std_ddof1']:.4f} & [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] & {diff_str} & {r['verdict']} \\\\\n"
tex_p1 += r"""\bottomrule
\end{tabular}
\end{table*}
"""
with open("tables/tab_p1_300epoch.tex", "w") as f:
    f.write(tex_p1)
print("\n[+] Wrote tables/tab_p1_300epoch.tex")
