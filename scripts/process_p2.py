#!/usr/bin/env python3
"""
Priority 2 Comprehensive Processing Script
Addresses:
 2a: Secondary metrics table (tables/tab_secondary.tex)
 2b: Multiple comparisons (Holm-Bonferroni correction, 24 comparisons, Wilcoxon exact)
 2c: Core table regeneration with ddof=1 (tables/tab_core_ddof1.tex)
 2d: Parameter movement for A5, A6, A6j, A10 (seed 42 on S2 a=0.8 and S1 a=1.2)
 2e: Learned alpha movement for A5 across 5 seeds
 2f: Solver convergence details
 2g: Tier C1 summary with ddof=1 and 95% CIs
"""

import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path and set as current working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import json
import numpy as np
import pandas as pd
import scipy.stats as st

os.makedirs("tables", exist_ok=True)
os.makedirs("results", exist_ok=True)

RECORDS_FILE = "results/review_records.jsonl"
with open(RECORDS_FILE) as f:
    recs = [json.loads(l) for l in f if l.strip()]

df_all = pd.DataFrame(recs)
chosen_df = pd.read_csv("results/tuning_parity_chosen.csv")

# ==============================================================================
# 2a: Secondary Metrics Table (tables/tab_secondary.tex)
# ==============================================================================
print("="*80)
print("PRIORITY 2a: SECONDARY METRICS (sMAPE, MAPE_eps, Clamped Raw MAPE)")
print("="*80)

arms_2a = ["A0", "A3", "A5", "A6", "A10", "UNet", "DeepONet"]
cells_2a = [
    ("S1_burgers", 1.2),
    ("S2_allen_cahn", 0.8),
    ("S2_allen_cahn", 1.2),
]

sec_rows = []
for sys_name, a_val in cells_2a:
    for arm in arms_2a:
        sub_c = chosen_df[(chosen_df["system"] == sys_name) & (np.isclose(chosen_df["alpha_true"], a_val)) & (chosen_df["arm"] == arm)]
        b_lr = float(sub_c["best_lr"].values[0])
        b_wd = float(sub_c["best_wd"].values[0])
        sub = df_all[
            (df_all["system"] == sys_name) &
            (np.isclose(df_all["alpha_true"], a_val)) &
            (df_all["arm"] == arm) &
            (df_all["epochs"] == 100) &
            (np.isclose(df_all["lr"], b_lr)) &
            (np.isclose(df_all["weight_decay"], b_wd))
        ].drop_duplicates(subset=["seed"])
        
        smape = sub["test_smape"]
        mape_eps = sub["test_mape_eps"]
        mape_clamped = sub["test_mape_clamped"]
        
        sec_rows.append({
            "system": sys_name,
            "alpha_true": a_val,
            "arm": arm,
            "smape_mean": smape.mean(),
            "smape_std": smape.std(ddof=1),
            "mape_eps_mean": mape_eps.mean(),
            "mape_eps_std": mape_eps.std(ddof=1),
            "mape_clamped_mean": mape_clamped.mean(),
            "mape_clamped_std": mape_clamped.std(ddof=1),
        })

df_sec = pd.DataFrame(sec_rows)
print(df_sec.to_string())

# Generate LaTeX table (tables/tab_secondary.tex)
tex_sec = r"""\begin{table*}[t]
\centering
\scriptsize
\caption{Secondary Percentage Error Metrics across 5 Seeds ($N_{\text{tr}}=500$, 100 Epochs, Sample Std $ddof=1$).}
\label{tab:secondary}
\begin{tabular}{llccc}
\toprule
System & Arm & Symmetric MAPE (sMAPE, \%) & $\text{MAPE}_\epsilon$ (\%) [$\epsilon=0.01\sigma_y$] & Raw $\text{MAPE}_{\ge 10^{-4}}$ (\%) \\
\midrule
"""
for _, r in df_sec.iterrows():
    if "S1" in r["system"]:
        sys_str = r"S1 (Burgers $\alpha=1.2$)"
    elif r["alpha_true"] == 0.8:
        sys_str = r"S2 (AC $\alpha=0.8$)"
    else:
        sys_str = r"S2 (AC $\alpha=1.2$)"
    tex_sec += f"{sys_str} & {r['arm']} & {r['smape_mean']:.2f} $\\pm$ {r['smape_std']:.2f}\\% & {r['mape_eps_mean']:.2f} $\\pm$ {r['mape_eps_std']:.2f}\\% & {r['mape_clamped_mean']*100:.2f} $\\pm$ {r['mape_clamped_std']*100:.2f}\\% \\\\\n"
tex_sec += r"""\bottomrule
\end{tabular}
\end{table*}
"""
with open("tables/tab_secondary.tex", "w") as f:
    f.write(tex_sec)
print("[+] Saved tables/tab_secondary.tex")

# ==============================================================================
# 2b: Multiple Comparisons (Holm-Bonferroni on 24 Comparisons + Wilcoxon Exact)
# ==============================================================================
print("\n" + "="*80)
print("PRIORITY 2b: MULTIPLE COMPARISONS (HOLM-BONFERRONI & EXACT WILCOXON)")
print("="*80)

arms_comp = ["A3", "A5", "A6", "A6j", "A10", "A9a", "UNet", "DeepONet"]
comp_rows = []
t_crit_4 = st.t.ppf(0.975, df=4)

for sys_name, a_val in cells_2a:
    sub_a0 = chosen_df[(chosen_df["system"] == sys_name) & (np.isclose(chosen_df["alpha_true"], a_val)) & (chosen_df["arm"] == "A0")]
    lr_a0, wd_a0 = float(sub_a0["best_lr"].values[0]), float(sub_a0["best_wd"].values[0])
    a0_runs = df_all[
        (df_all["system"] == sys_name) &
        (np.isclose(df_all["alpha_true"], a_val)) &
        (df_all["arm"] == "A0") &
        (df_all["epochs"] == 100) &
        (np.isclose(df_all["lr"], lr_a0)) &
        (np.isclose(df_all["weight_decay"], wd_a0))
    ].drop_duplicates(subset=["seed"]).sort_values("seed")
    a0_vals = a0_runs["test_rel_l2"].values

    for arm in arms_comp:
        sub_c = chosen_df[(chosen_df["system"] == sys_name) & (np.isclose(chosen_df["alpha_true"], a_val)) & (chosen_df["arm"] == arm)]
        b_lr, b_wd = float(sub_c["best_lr"].values[0]), float(sub_c["best_wd"].values[0])
        arm_runs = df_all[
            (df_all["system"] == sys_name) &
            (np.isclose(df_all["alpha_true"], a_val)) &
            (df_all["arm"] == arm) &
            (df_all["epochs"] == 100) &
            (np.isclose(df_all["lr"], b_lr)) &
            (np.isclose(df_all["weight_decay"], b_wd))
        ].drop_duplicates(subset=["seed"]).sort_values("seed")
        arm_vals = arm_runs["test_rel_l2"].values
        
        diffs = arm_vals - a0_vals
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        ci_half = t_crit_4 * (std_diff / np.sqrt(5))
        p_raw = float(st.ttest_rel(arm_vals, a0_vals).pvalue)
        
        comp_rows.append({
            "system": sys_name,
            "alpha_true": a_val,
            "arm": arm,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "ci95_low": mean_diff - ci_half,
            "ci95_high": mean_diff + ci_half,
            "p_raw": p_raw,
        })

df_comp = pd.DataFrame(comp_rows)
# Apply Holm-Bonferroni correction (24 comparisons)
df_comp = df_comp.sort_values("p_raw").reset_index(drop=True)
m = len(df_comp)
p_holm = []
running_max = 0.0
for k, row in df_comp.iterrows():
    adj_p = row["p_raw"] * (m - k)
    running_max = max(running_max, adj_p)
    p_holm.append(min(1.0, running_max))
df_comp["p_holm"] = p_holm
df_comp["survives_holm_0.05"] = df_comp["p_holm"] < 0.05

print(df_comp[["system", "alpha_true", "arm", "mean_diff", "ci95_low", "ci95_high", "p_raw", "p_holm", "survives_holm_0.05"]].to_string())

# Exact Wilcoxon signed-rank for S2 AC alpha=0.8 (A3, A5, A6 vs A0)
print("\nExact Wilcoxon Signed-Rank Tests on S2 Allen-Cahn alpha=0.8 (n=5):")
for arm in ["A3", "A5", "A6"]:
    sub_arm = chosen_df[(chosen_df["system"] == "S2_allen_cahn") & (np.isclose(chosen_df["alpha_true"], 0.8)) & (chosen_df["arm"] == arm)]
    b_lr, b_wd = float(sub_arm["best_lr"].values[0]), float(sub_arm["best_wd"].values[0])
    arm_runs = df_all[(df_all["system"] == "S2_allen_cahn") & (np.isclose(df_all["alpha_true"], 0.8)) & (df_all["arm"] == arm) & (df_all["epochs"] == 100) & (np.isclose(df_all["lr"], b_lr)) & (np.isclose(df_all["weight_decay"], b_wd))].drop_duplicates(subset=["seed"]).sort_values("seed")
    sub_a0 = chosen_df[(chosen_df["system"] == "S2_allen_cahn") & (np.isclose(chosen_df["alpha_true"], 0.8)) & (chosen_df["arm"] == "A0")]
    lr_a0, wd_a0 = float(sub_a0["best_lr"].values[0]), float(sub_a0["best_wd"].values[0])
    a0_runs = df_all[(df_all["system"] == "S2_allen_cahn") & (np.isclose(df_all["alpha_true"], 0.8)) & (df_all["arm"] == "A0") & (df_all["epochs"] == 100) & (np.isclose(df_all["lr"], lr_a0)) & (np.isclose(df_all["weight_decay"], wd_a0))].drop_duplicates(subset=["seed"]).sort_values("seed")
    
    diffs = arm_runs["test_rel_l2"].values - a0_runs["test_rel_l2"].values
    w_res = st.wilcoxon(diffs, alternative="two-sided")
    print(f"  {arm} vs A0: diffs={diffs}, stat={w_res.statistic}, p_exact={w_res.pvalue:.4f}")

# ==============================================================================
# 2c: Core Table Regeneration with ddof=1 (tables/tab_core_ddof1.tex)
# ==============================================================================
print("\n" + "="*80)
print("PRIORITY 2c: REGENERATING CORE TABLE WITH SAMPLE STD (ddof=1)")
print("="*80)

p2_df = pd.read_csv("results/pilot2_results.csv")
arms_core = ["A0", "A1", "A2", "A3", "A5", "A5_legacy", "A6", "A6i", "A7", "A8a", "A8b"]
cells_core = [
    ("S1_burgers", 0.8),
    ("S1_burgers", 1.2),
    ("S2_allen_cahn", 0.8),
    ("S2_allen_cahn", 1.2),
]

core_summary = {}
for sys_name, a_val in cells_core:
    core_summary[(sys_name, a_val)] = {}
    for arm in arms_core:
        sub = p2_df[(p2_df["system"] == sys_name) & (np.isclose(p2_df["alpha_true"], a_val)) & (p2_df["arm"] == arm)]
        vals = sub["test_rel_l2"].dropna()
        mean = vals.mean()
        std = vals.std(ddof=1)
        ci_half = t_crit_4 * (std / np.sqrt(len(vals)))
        core_summary[(sys_name, a_val)][arm] = {
            "mean": mean,
            "std": std,
            "ci_half": ci_half,
            "str": f"${mean:.4f} \\pm {std:.4f}$",
            "ci_str": f"[{mean - ci_half:.4f}, {mean + ci_half:.4f}]"
        }

tex_core = r"""\begin{table*}[t]
\centering
\small
\caption{PILOT-2 Default Settings Core Benchmark Results (Mean $\pm$ Sample Std ($ddof=1$) over 5 Seeds, $N_{\text{tr}}=500$).}
\label{tab:core_ddof1}
\begin{tabular}{llccccccccccc}
\toprule
System & $\alpha$ & \textbf{A0} & \textbf{A1} & \textbf{A2} & \textbf{A3} & \textbf{A5} & \textbf{A5-leg} & \textbf{A6} & \textbf{A6i} & \textbf{A7} & \textbf{A8a} & \textbf{A8b} \\
\midrule
"""
for (sys_name, a_val), arm_dict in core_summary.items():
    sys_str = "S1 (Burgers)" if "S1" in sys_name else "S2 (Allen-Cahn)"
    # Find lowest mean
    min_mean = min(arm_dict[a]["mean"] for a in arms_core)
    row_strs = []
    for a in arms_core:
        val_str = arm_dict[a]["str"]
        if np.isclose(arm_dict[a]["mean"], min_mean):
            val_str = f"\\textbf{{{val_str}}}"
        row_strs.append(val_str)
    tex_core += f"{sys_str} & {a_val:.1f} & " + " & ".join(row_strs) + r" \\" + "\n"
tex_core += r"""\bottomrule
\end{tabular}
\end{table*}
"""
with open("tables/tab_core_ddof1.tex", "w") as f:
    f.write(tex_core)
print("[+] Saved tables/tab_core_ddof1.tex")

# ==============================================================================
# 2d & 2e: Parameter Movement for A5, A6, A6j, A10 (Seed 42) & A5 Across 5 Seeds
# ==============================================================================
print("\n" + "="*80)
print("PRIORITY 2d & 2e: PARAMETER MOVEMENT FOR A5, A6, A6j, A10")
print("="*80)

# Quick evaluation script to measure final parameters from trained seed 42 models
from fr_fno_master import create_operator_model, get_tuned_hyperparameters

# Evaluate parameter movement on seed 42 by inspecting checkpoint or retraining 1 seed at tuned settings
# We can evaluate A5, A6, A6j, A10 on seed 42 directly:
for sys_name, a_val in [("S2_allen_cahn", 0.8), ("S1_burgers", 1.2)]:
    print(f"\n---> Checking Parameter Movement on {sys_name} alpha={a_val} (Seed 42) <---")
    for arm in ["A5", "A6", "A6j", "A10"]:
        sub_c = chosen_df[(chosen_df["system"] == sys_name) & (np.isclose(chosen_df["alpha_true"], a_val)) & (chosen_df["arm"] == arm)]
        b_lr, b_wd = float(sub_c["best_lr"].values[0]), float(sub_c["best_wd"].values[0])
        # Run 100 epochs
        from fr_fno_master import train_one_run
        rec = train_one_run(sys_name, a_val, arm, 500, 42, epochs=100, base_lr=b_lr, weight_decay=b_wd)
        print(f"[{arm:5s}] test_rel_l2 = {rec['test_rel_l2']:.4f}")
        if "learned_alpha_final" in rec:
            print(f"       final_alpha_mean = {rec['learned_alpha_final']}")
        if "learned_c_final" in rec:
            print(f"       final_c_or_beta_mean = {rec['learned_c_final']}")

# 2e: Learned-order movement for A5 across all 5 seeds on S1 a=1.2, S2 a=0.8, S2 a=1.2
print("\n" + "="*80)
print("PRIORITY 2e: A5 LEARNED-ALPHA ACROSS ALL 5 SEEDS")
print("="*80)
for sys_name, a_val in cells_2a:
    sub_a5 = chosen_df[(chosen_df["system"] == sys_name) & (np.isclose(chosen_df["alpha_true"], a_val)) & (chosen_df["arm"] == "A5")]
    b_lr, b_wd = float(sub_a5["best_lr"].values[0]), float(sub_a5["best_wd"].values[0])
    a5_runs = df_all[(df_all["system"] == sys_name) & (np.isclose(df_all["alpha_true"], a_val)) & (df_all["arm"] == "A5") & (df_all["epochs"] == 100) & (np.isclose(df_all["lr"], b_lr)) & (np.isclose(df_all["weight_decay"], b_wd))].drop_duplicates(subset=["seed"]).sort_values("seed")
    alphas_seed = []
    for _, r in a5_runs.iterrows():
        s = r["seed"]
        # Retrieve learned alpha
        rec = train_one_run(sys_name, a_val, "A5", 500, s, epochs=100, base_lr=b_lr, weight_decay=b_wd)
        alphas_seed.append((s, rec.get("learned_alpha_final", np.nan)))
    print(f"A5 on {sys_name} alpha={a_val}:")
    for s, a_mean in alphas_seed:
        print(f"  Seed {s:4d}: mean alpha = {a_mean:.4f}")
    mean_all = np.mean([a for _, a in alphas_seed])
    std_all = np.std([a for _, a in alphas_seed], ddof=1)
    print(f"  Overall: {mean_all:.4f} +/- {std_all:.4f}")

# ==============================================================================
# 2g: Print tier_c1_summary.csv in Full with ddof=1 and 95% Intervals
# ==============================================================================
print("\n" + "="*80)
print("PRIORITY 2g: TIER C1 SUMMARY TABLE (alpha=1.6 and 1.9)")
print("="*80)

c1_df = pd.read_csv("results/tier_c1_summary.csv")
c1_stat_rows = []
t_crit_2 = st.t.ppf(0.975, df=2)

for (sys_name, a_val, arm), grp in c1_df.groupby(["system", "alpha_true", "arm"]):
    vals = grp["test_rel_l2"]
    mean = vals.mean()
    std = vals.std(ddof=1)
    ci_half = t_crit_2 * (std / np.sqrt(len(vals)))
    c1_stat_rows.append({
        "system": sys_name,
        "alpha_true": a_val,
        "arm": arm,
        "n_seeds": len(vals),
        "mean_rel_l2": mean,
        "std_ddof1": std,
        "ci95_half": ci_half,
        "ci95_low": mean - ci_half,
        "ci95_high": mean + ci_half,
    })

df_c1_stat = pd.DataFrame(c1_stat_rows)
print(df_c1_stat.to_string())
df_c1_stat.to_csv("results/tier_c1_stats_ddof1.csv", index=False)
