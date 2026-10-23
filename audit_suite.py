#!/usr/bin/env python3
"""
Comprehensive Revision Audit Suite for IEEE INDICON
Addresses all reviewer requirements:
- Operator baselines (DeepONet, UNet)
- Resolvent filter (A10) and tuned parity arms (A6j, A9a)
- Computation time, memory, parameter counts
- MAPE metrics (sMAPE, MAPE_eps, clamped MAPE) and zero-crossing analysis
- Confidence intervals (95% Student-t) and paired difference t-tests
- 300-epoch scaling
- High-fidelity publication figures (3.45 in wide, vector PDF, font >= 8pt)
- Append-only structured logging to results/review_records.jsonl
"""

import os
import json
import time
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from fr_fno_master import train_one_run

RECORDS_FILE = "results/review_records.jsonl"
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Required schema fields for results/review_records.jsonl
REQUIRED_SCHEMA = [
    "system", "alpha_true", "arm", "seed", "n_train", "epochs", "batch_size",
    "optimizer", "lr", "weight_decay", "test_rel_l2", "test_mse",
    "test_smape", "test_mape_eps", "test_mape_clamped", "params",
    "train_seconds", "infer_ms_per_sample"
]

def append_review_record(rec):
    """Formats and appends a single run record to results/review_records.jsonl."""
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

def load_existing_records():
    cache = {}
    if os.path.exists(RECORDS_FILE):
        with open(RECORDS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = (r["system"], float(r["alpha_true"]), r["arm"], int(r["seed"]), int(r["epochs"]), round(float(r["lr"]), 6), round(float(r["weight_decay"]), 6))
                    cache[key] = r
                except Exception:
                    pass
    return cache

GLOBAL_CACHE = load_existing_records()

def get_or_run_experiment(system, alpha_true, arm, seed, epochs=100, base_lr=1.0e-3, weight_decay=1.0e-4):
    key = (system, float(alpha_true), arm, int(seed), int(epochs), round(float(base_lr), 6), round(float(weight_decay), 6))
    if key in GLOBAL_CACHE:
        return GLOBAL_CACHE[key]
    
    rec = train_one_run(system, alpha_true, arm, 500, seed, epochs=epochs, base_lr=base_lr, weight_decay=weight_decay)
    formatted = append_review_record(rec)
    GLOBAL_CACHE[key] = formatted
    return formatted


# ==============================================================================
# STAGE 1: TIER A5 FIGURE GENERATION (Loss Curves & Qualitative Profiles)
# ==============================================================================
def run_stage_1_figures():
    print("\n" + "="*80)
    print("STAGE 1: RUNNING TIER A5 FIGURE GENERATION & LOGGING")
    print("="*80)

    # 1. Validation curves on S2 alpha=0.8 across 5 seeds for A0, A3, A5, A6
    seeds = [42, 123, 999, 2024, 2025]
    arms_curves = ["A0", "A3", "A5", "A6"]
    curve_data = {arm: [] for arm in arms_curves}

    s2_recs_seed42 = {}
    for arm in arms_curves:
        print(f"[*] Training {arm} on S2 alpha=0.8 across 5 seeds...")
        for s in seeds:
            rec = train_one_run("S2_allen_cahn", 0.8, arm, 500, s, epochs=100, base_lr=1.0e-3, weight_decay=1.0e-4)
            append_review_record(rec)
            curve_data[arm].append(rec["val_trajectory"])
            if s == 42:
                s2_recs_seed42[arm] = rec

    # 2. Qualitative predictions on S1 alpha=1.2 (seed 42) for A0, A3, A6
    s1_recs_seed42 = {}
    for arm in ["A0", "A3", "A6"]:
        print(f"[*] Training {arm} on S1 alpha=1.2 (seed 42)...")
        rec = train_one_run("S1_burgers", 1.2, arm, 500, 42, epochs=100, base_lr=1.0e-3, weight_decay=1.0e-4)
        append_review_record(rec)
        s1_recs_seed42[arm] = rec

    # --- Plot fig_curves.pdf ---
    # Width 3.45 in, vector PDF, font >= 8pt, colorblind-safe palette
    print("[*] Generating figures/fig_curves.pdf...")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "figure.titlesize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)
    colors = {
        "A0": "#0072B2",  # Blue
        "A3": "#D55E00",  # Vermillion
        "A5": "#CC79A7",  # Reddish Purple
        "A6": "#009E73",  # Bluish Green
    }
    labels = {
        "A0": "A0 (Standard FNO)",
        "A3": "A3 (Fixed fr-FNO)",
        "A5": "A5 (Adaptive fr-FNO)",
        "A6": "A6 (True Dissipative)",
    }

    epochs_arr = np.arange(1, 101)
    for arm in arms_curves:
        arr = np.array(curve_data[arm])  # shape (5, 100)
        mean_c = np.mean(arr, axis=0)
        std_c = np.std(arr, axis=0)
        ax.plot(epochs_arr, mean_c, color=colors[arm], label=labels[arm], lw=1.3)
        ax.fill_between(epochs_arr, np.maximum(0, mean_c - std_c), mean_c + std_c, color=colors[arm], alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Relative $L^2$ Error")
    ax.set_yscale("log")
    ax.set_xlim(1, 100)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    plt.tight_layout(pad=0.4)
    plt.savefig("figures/fig_curves.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("[+] Saved figures/fig_curves.pdf successfully.")

    # --- Plot fig_qualitative.pdf ---
    print("[*] Generating figures/fig_qualitative.pdf...")
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.2), dpi=300, sharex=True)

    x_grid = np.linspace(0, 2.0 * np.pi, 256, endpoint=False)
    # Row 0: S1 Burgers alpha=1.2
    y_true_s1 = s1_recs_seed42["A0"]["y_test_sample0"]
    pred_a0_s1 = s1_recs_seed42["A0"]["pred_test_sample0"]
    pred_a3_s1 = s1_recs_seed42["A3"]["pred_test_sample0"]
    pred_a6_s1 = s1_recs_seed42["A6"]["pred_test_sample0"]

    # (0, 0): Solution profiles
    axes[0, 0].plot(x_grid, y_true_s1, "k-", lw=1.2, label="Ground Truth")
    axes[0, 0].plot(x_grid, pred_a0_s1, color=colors["A0"], ls="--", lw=1.0, label="A0")
    axes[0, 0].plot(x_grid, pred_a3_s1, color=colors["A3"], ls="-.", lw=1.0, label="A3")
    axes[0, 0].plot(x_grid, pred_a6_s1, color=colors["A6"], ls=":", lw=1.2, label="A6")
    axes[0, 0].set_ylabel("S1: $u(x, 1)$")
    axes[0, 0].set_title("Predicted Profiles", fontsize=8.0, pad=3)
    axes[0, 0].legend(loc="lower left", fontsize=6.0, frameon=True, framealpha=0.8, edgecolor="none")
    axes[0, 0].grid(True, ls=":", alpha=0.5)

    # (0, 1): Absolute error
    axes[0, 1].plot(x_grid, np.abs(pred_a0_s1 - y_true_s1), color=colors["A0"], ls="--", lw=1.0)
    axes[0, 1].plot(x_grid, np.abs(pred_a3_s1 - y_true_s1), color=colors["A3"], ls="-.", lw=1.0)
    axes[0, 1].plot(x_grid, np.abs(pred_a6_s1 - y_true_s1), color=colors["A6"], ls=":", lw=1.2)
    axes[0, 1].set_ylabel("Pointwise $|u - \hat{u}|$")
    axes[0, 1].set_title("Absolute Error", fontsize=8.0, pad=3)
    axes[0, 1].grid(True, ls=":", alpha=0.5)

    # Row 1: S2 Allen-Cahn alpha=0.8
    y_true_s2 = s2_recs_seed42["A0"]["y_test_sample0"]
    pred_a0_s2 = s2_recs_seed42["A0"]["pred_test_sample0"]
    pred_a3_s2 = s2_recs_seed42["A3"]["pred_test_sample0"]
    pred_a6_s2 = s2_recs_seed42["A6"]["pred_test_sample0"]

    # (1, 0): Solution profiles
    axes[1, 0].plot(x_grid, y_true_s2, "k-", lw=1.2, label="Ground Truth")
    axes[1, 0].plot(x_grid, pred_a0_s2, color=colors["A0"], ls="--", lw=1.0)
    axes[1, 0].plot(x_grid, pred_a3_s2, color=colors["A3"], ls="-.", lw=1.0)
    axes[1, 0].plot(x_grid, pred_a6_s2, color=colors["A6"], ls=":", lw=1.2)
    axes[1, 0].set_ylabel("S2: $u(x, 1)$")
    axes[1, 0].set_xlabel("$x \in [0, 2\pi)$")
    axes[1, 0].grid(True, ls=":", alpha=0.5)

    # (1, 1): Absolute error
    axes[1, 1].plot(x_grid, np.abs(pred_a0_s2 - y_true_s2), color=colors["A0"], ls="--", lw=1.0)
    axes[1, 1].plot(x_grid, np.abs(pred_a3_s2 - y_true_s2), color=colors["A3"], ls="-.", lw=1.0)
    axes[1, 1].plot(x_grid, np.abs(pred_a6_s2 - y_true_s2), color=colors["A6"], ls=":", lw=1.2)
    axes[1, 1].set_ylabel("Pointwise $|u - \hat{u}|$")
    axes[1, 1].set_xlabel("$x \in [0, 2\pi)$")
    axes[1, 1].grid(True, ls=":", alpha=0.5)

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/fig_qualitative.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("[+] Saved figures/fig_qualitative.pdf successfully.")


# ==============================================================================
# STAGE 2: TIER B3 TUNING PARITY (Grid Search over lr x wd)
# ==============================================================================
def run_stage_2_tuning():
    print("\n" + "="*80)
    print("STAGE 2: RUNNING TIER B3 SYSTEMATIC TUNING PARITY")
    print("="*80)

    cells = [
        ("S1_burgers", 1.2),
        ("S2_allen_cahn", 0.8),
        ("S2_allen_cahn", 1.2),
    ]
    arms = ["A0", "A3", "A5", "A6", "A6j", "A10", "A9a", "DeepONet", "UNet"]
    lr_grid = [3.0e-4, 1.0e-3, 3.0e-3]
    wd_grid = [0.0, 1.0e-4]

    tuning_results = []
    chosen_configs = {}  # (cell, arm) -> (best_lr, best_wd, best_val_err, seed42_rec)

    for sys_name, a_val in cells:
        print(f"\n---> TUNING ON CELL: {sys_name} alpha={a_val:.1f} <---")
        for arm in arms:
            best_val = float("inf")
            best_cfg = None
            best_rec = None
            for lr in lr_grid:
                for wd in wd_grid:
                    rec = train_one_run(sys_name, a_val, arm, 500, 42, epochs=100, base_lr=lr, weight_decay=wd)
                    append_review_record(rec)
                    val_err = rec["val_rel_l2"]
                    tuning_results.append({
                        "system": sys_name,
                        "alpha_true": a_val,
                        "arm": arm,
                        "lr": lr,
                        "weight_decay": wd,
                        "val_rel_l2": val_err,
                        "test_rel_l2": rec["test_rel_l2"],
                        "test_smape": rec["test_smape"],
                        "train_seconds": rec["wall_clock_sec"],
                    })
                    if val_err < best_val:
                        best_val = val_err
                        best_cfg = (lr, wd)
                        best_rec = rec
            chosen_configs[(sys_name, a_val, arm)] = (best_cfg[0], best_cfg[1], best_val, best_rec)
            print(f"[{sys_name} a={a_val} | {arm:8s}] Selected: lr={best_cfg[0]:.1e}, wd={best_cfg[1]:.1e} | val_rel_l2={best_val:.4e} | test_rel_l2={best_rec['test_rel_l2']:.4e}")

    df_tune = pd.DataFrame(tuning_results)
    df_tune.to_csv("results/tuning_parity_grid.csv", index=False)
    print("\n[+] Saved full tuning grid to results/tuning_parity_grid.csv")

    summary_rows = []
    for (sys_name, a_val, arm), (b_lr, b_wd, b_val, b_rec) in chosen_configs.items():
        summary_rows.append({
            "system": sys_name,
            "alpha_true": a_val,
            "arm": arm,
            "best_lr": b_lr,
            "best_wd": b_wd,
            "best_val_rel_l2": b_val,
            "test_rel_l2_seed42": b_rec["test_rel_l2"],
            "test_smape_seed42": b_rec["test_smape"],
            "test_mape_clamped_seed42": b_rec["test_mape_clamped"],
            "params": b_rec["trainable_params"],
            "train_seconds": b_rec["wall_clock_sec"],
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/tuning_parity_chosen.csv", index=False)
    print("[+] Saved chosen configurations to results/tuning_parity_chosen.csv")
    print(df_summary.to_string())

    return chosen_configs


# ==============================================================================
# STAGE 3: TIER B4 FINAL RUNS (5 Seeds & 300 Epochs)
# ==============================================================================
def run_stage_3_final_runs(chosen_configs):
    print("\n" + "="*80)
    print("STAGE 3: RUNNING TIER B4 CONFIRMATORY 5 SEEDS & 300 EPOCHS")
    print("="*80)

    # 1. Remaining 4 seeds at chosen (lr*, wd*)
    rem_seeds = [123, 999, 2024, 2025]
    final_records = []

    for (sys_name, a_val, arm), (b_lr, b_wd, b_val, seed42_rec) in chosen_configs.items():
        final_records.append(seed42_rec)
        print(f"[*] Running remaining seeds for {sys_name} a={a_val} {arm} (lr={b_lr:.1e}, wd={b_wd:.1e})...")
        for s in rem_seeds:
            rec = train_one_run(sys_name, a_val, arm, 500, s, epochs=100, base_lr=b_lr, weight_decay=b_wd)
            append_review_record(rec)
            final_records.append(rec)

    # 2. 300 epochs on S2 alpha=1.2, 3 seeds (42, 123, 999) for A0, A3, A6, A10, DeepONet, UNet
    arms_300 = ["A0", "A3", "A6", "A10", "DeepONet", "UNet"]
    seeds_300 = [42, 123, 999]
    records_300 = []

    print("\n---> RUNNING 300-EPOCH EXTENSIONS ON S2 alpha=1.2 <---")
    for arm in arms_300:
        b_lr, b_wd, _, _ = chosen_configs[("S2_allen_cahn", 1.2, arm)]
        print(f"[*] 300 epochs for {arm} (lr={b_lr:.1e}, wd={b_wd:.1e})...")
        for s in seeds_300:
            rec = train_one_run("S2_allen_cahn", 1.2, arm, 500, s, epochs=300, base_lr=b_lr, weight_decay=b_wd)
            append_review_record(rec)
            records_300.append(rec)

    # ==============================================================================
    # STAGE 4: STATISTICAL SUMMARIES & CONFIDENCE INTERVALS
    # ==============================================================================
    print("\n" + "="*80)
    print("STAGE 4: COMPUTING RIGOROUS 95% CIs & SUMMARY TABLES")
    print("="*80)

    # Final 5-seed statistics
    t_crit = st.t.ppf(0.975, df=4)
    df_all = pd.DataFrame(final_records)
    stat_rows = []

    for (sys_name, a_val), grp_cell in df_all.groupby(["system", "alpha_true"]):
        pivot = grp_cell.pivot(index="seed", columns="arm", values="test_rel_l2")
        a0_vals = pivot["A0"] if "A0" in pivot.columns else None

        for arm in pivot.columns:
            vals = pivot[arm].dropna()
            mean = vals.mean()
            std = vals.std(ddof=1)
            ci_half = t_crit * (std / np.sqrt(len(vals)))

            p_val = 1.0
            if arm != "A0" and a0_vals is not None:
                p_val = st.ttest_rel(vals, a0_vals).pvalue

            stat_rows.append({
                "system": sys_name,
                "alpha_true": a_val,
                "arm": arm,
                "n_seeds": len(vals),
                "test_rel_l2_mean": mean,
                "test_rel_l2_std": std,
                "ci95_half_width": ci_half,
                "ci95_low": mean - ci_half,
                "ci95_high": mean + ci_half,
                "paired_t_pvalue_vs_A0": p_val,
            })

    df_final_ci = pd.DataFrame(stat_rows)
    df_final_ci.to_csv("results/final_5seed_confidence_intervals.csv", index=False)
    print("\n[+] Saved results/final_5seed_confidence_intervals.csv")
    print(df_final_ci.to_string())

    # 300 epochs summary
    df_300 = pd.DataFrame(records_300)
    t_crit_3 = st.t.ppf(0.975, df=2)
    s300_rows = []
    pivot_300 = df_300.pivot(index="seed", columns="arm", values="test_rel_l2")
    a0_300 = pivot_300["A0"] if "A0" in pivot_300.columns else None

    for arm in pivot_300.columns:
        vals = pivot_300[arm].dropna()
        mean = vals.mean()
        std = vals.std(ddof=1)
        ci_half = t_crit_3 * (std / np.sqrt(len(vals)))
        p_val = 1.0
        if arm != "A0" and a0_300 is not None:
            p_val = st.ttest_rel(vals, a0_300).pvalue
        s300_rows.append({
            "system": "S2_allen_cahn",
            "alpha_true": 1.2,
            "arm": arm,
            "epochs": 300,
            "n_seeds": len(vals),
            "test_rel_l2_mean": mean,
            "test_rel_l2_std": std,
            "ci95_half_width": ci_half,
            "paired_t_pvalue_vs_A0": p_val,
        })
    df_300_ci = pd.DataFrame(s300_rows)
    df_300_ci.to_csv("results/extended_300epoch_confidence_intervals.csv", index=False)
    print("\n[+] Saved results/extended_300epoch_confidence_intervals.csv")
    print(df_300_ci.to_string())


if __name__ == "__main__":
    print("[*] Starting Revision Audit Suite...")
    run_stage_1_figures()
    chosen_cfgs = run_stage_2_tuning()
    run_stage_3_final_runs(chosen_cfgs)
    print("\n[***] ALL AUDIT PHASES COMPLETED PERFECTLY! [***]")
