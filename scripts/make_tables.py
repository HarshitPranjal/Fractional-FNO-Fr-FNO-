import json
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path and set as current working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import re
import numpy as np
import scipy.stats as stats

JSONL_PATH = "results/experiment_records.jsonl"
T1_TEX_PATH = "tables/T1_main_results.tex"
T4_TEX_PATH = "tables/T4_alpha_recovery.tex"
MAIN_TEX_PATH = "main.tex"

def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        print(f"[!] {path} not found.")
        return records
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError:
                print(f"[!] Warning: skipped malformed JSON at line {line_no}")
    return records

def generate_table_1_rel_l2(records):
    """
    Generates Table I: Relative L2 test error across systems, alphas, train_sizes, and arms.
    """
    groups = {}
    for r in records:
        key = (r["system"], float(r["alpha_true"]), int(r["train_size"]))
        arm = r["arm"]
        if key not in groups:
            groups[key] = {}
        if arm not in groups[key]:
            groups[key][arm] = []
        if not np.isnan(r.get("test_rel_l2", float("nan"))):
            groups[key][arm].append(r["test_rel_l2"])
    
    summary = {}
    for key, arm_dict in sorted(groups.items()):
        summary[key] = {}
        for arm, vals in arm_dict.items():
            if len(vals) > 0:
                summary[key][arm] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                    "str": f"{np.mean(vals):.4f} \\pm {np.std(vals):.4f}"
                }
    return summary

def generate_table_2_alpha_eff(records):
    """
    Generates Table II: Effective alpha recovery (alpha_eff) on S3 Diffusion across 5 seeds.
    Computes per-alpha mean ± std, and over all 20 runs: Spearman rho, p-value, and MAE.
    """
    s3_records = [r for r in records if r["system"] == "S3_diffusion"]
    arms = sorted(list(set(r["arm"] for r in s3_records)))
    
    table2_data = {}
    for arm in arms:
        arm_recs = [r for r in s3_records if r["arm"] == arm]
        alphas_true = np.array([r["alpha_true"] for r in arm_recs])
        alphas_eff = np.array([r.get("alpha_eff", float("nan")) for r in arm_recs])
        
        valid = ~np.isnan(alphas_eff)
        a_true_v = alphas_true[valid]
        a_eff_v = alphas_eff[valid]
        
        per_alpha = {}
        for a_val in sorted(list(set(a_true_v))):
            sub_eff = a_eff_v[a_true_v == a_val]
            per_alpha[a_val] = {
                "mean": float(np.mean(sub_eff)),
                "std": float(np.std(sub_eff)),
                "str": f"{np.mean(sub_eff):.4f} \\pm {np.std(sub_eff):.4f}"
            }
            
        rho, p_val = stats.spearmanr(a_true_v, a_eff_v) if len(a_eff_v) >= 4 else (float("nan"), float("nan"))
        mae = float(np.mean(np.abs(a_eff_v - a_true_v))) if len(a_eff_v) > 0 else float("nan")
        
        table2_data[arm] = {
            "per_alpha": per_alpha,
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "mae": mae,
            "n_runs": len(a_eff_v),
        }
    return table2_data

def diff_against_tex(t1_summary, t2_summary):
    print("=" * 84)
    print("DIFFING JSONL NUMBERS AGAINST SAVED LATEX TABLES")
    print("=" * 84)
    
    # 1. Diff Table I against tables/T1_main_results.tex
    if os.path.exists(T1_TEX_PATH):
        with open(T1_TEX_PATH, "r") as f:
            tex_content = f.read()
            
        print(f"\n[+] Comparing Table I against {T1_TEX_PATH}:")
        mismatches_t1 = []
        matches_t1 = 0
        
        # Parse all data rows from T1_main_results.tex
        tex_rows = {}
        curr_sys = None
        for line in tex_content.splitlines():
            line_s = line.strip()
            if not line_s or line_s.startswith("%") or line_s.startswith("\\"):
                continue
            parts = [p.strip() for p in line_s.rstrip(r"\\").split("&")]
            if len(parts) >= 11:
                sys_str = parts[0].strip()
                if sys_str:
                    curr_sys = sys_str
                a_str = parts[1].strip()
                n_str = parts[2].strip()
                try:
                    a_val = float(a_str)
                    n_val = int(n_str)
                    cells = parts[3:11]
                    tex_rows[(curr_sys, a_val, n_val)] = cells
                except ValueError:
                    continue

        arm_names_tex = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        for (sys_name, a_val, tr_sz), arm_vals in t1_summary.items():
            # Match sys_name with curr_sys
            match_key = None
            for (t_sys, t_a, t_n), cells in tex_rows.items():
                if t_a == a_val and t_n == tr_sz:
                    if (sys_name == "S3_diffusion" and "Diffusion" in t_sys) or \
                       (sys_name == "S1_burgers" and "Burgers" in t_sys) or \
                       (sys_name == "S2_allen_cahn" and "Allen-Cahn" in t_sys):
                        match_key = (t_sys, t_a, t_n)
                        break

            if match_key:
                cells = tex_rows[match_key]
                for cell_idx, arm in enumerate(arm_names_tex):
                    if arm in arm_vals and cell_idx < len(cells):
                        jsonl_val = arm_vals[arm]["str"]
                        tex_val = cells[cell_idx].replace(r"\textbf{", "").replace("}", "").strip()
                        if tex_val == "---":
                            mismatches_t1.append(f"{sys_name} a={a_val:.1f} N={tr_sz} {arm}: JSONL has '{jsonl_val}' but T1.tex has '---' (missing in old 3-seed grid)")
                        elif tex_val != jsonl_val:
                            mismatches_t1.append(f"{sys_name} a={a_val:.1f} N={tr_sz} {arm}: JSONL='{jsonl_val}' vs T1.tex='{tex_val}'")
                        else:
                            matches_t1 += 1
        
        print(f"    • Total mismatches in Table I: {len(mismatches_t1)}")
        for m in mismatches_t1:
            print(f"      - {m}")
    else:
        print(f"[!] {T1_TEX_PATH} not found.")

    # 2. Diff Table II against tables/T4_alpha_recovery.tex
    if os.path.exists(T4_TEX_PATH):
        with open(T4_TEX_PATH, "r") as f:
            tex_t4 = f.read()
        print(f"\n[+] Comparing Table II (Alpha Recovery) against {T4_TEX_PATH}:")
        mismatches_t2 = []
        for arm, data in t2_summary.items():
            if arm in ["A5", "A6"]:
                rho_jsonl = data["spearman_rho"]
                mae_jsonl = data["mae"]
                # In T4.tex: A5 rho was -0.112, MAE was 0.5016 (raw parameters); A6 was -0.157, MAE was 0.5990
                mismatches_t2.append(f"Arm {arm}: JSONL probe alpha_eff has rho={rho_jsonl:.4f}, MAE={mae_jsonl:.4f} (5 seeds, N=20). T4.tex contains raw parameter stats from earlier 3-seed run.")
        for m in mismatches_t2:
            print(f"      - {m}")
            
    # 3. Check for main.tex
    if not os.path.exists(MAIN_TEX_PATH):
        print(f"\n[!] Note: '{MAIN_TEX_PATH}' does not exist on disk. Comparisons performed against 'tables/T1_main_results.tex' and 'tables/T4_alpha_recovery.tex'.")

if __name__ == "__main__":
    records = load_jsonl(JSONL_PATH)
    print(f"Loaded {len(records)} records from {JSONL_PATH}")
    
    t1 = generate_table_1_rel_l2(records)
    t2 = generate_table_2_alpha_eff(records)
    
    print("\n" + "=" * 84)
    print("TABLE I: RELATIVE L2 TEST ERROR (FROM JSONL)")
    print("=" * 84)
    for (sys_name, a_val, tr_sz), arm_dict in t1.items():
        row = f"{sys_name} a={a_val:.1f} N={tr_sz}: " + ", ".join([f"{arm}: {v['str']}" for arm, v in arm_dict.items()])
        print(row)
        
    print("\n" + "=" * 84)
    print("TABLE II: ALPHA_EFF RECOVERY & MAE OVER ALL 20 RUNS (FROM JSONL)")
    print("=" * 84)
    for arm, data in t2.items():
        print(f"Arm {arm} (N={data['n_runs']}): Spearman rho = {data['spearman_rho']:.4f} (p = {data['spearman_p']:.4e}), MAE = {data['mae']:.4f}")
        for a_val, stat in data["per_alpha"].items():
            print(f"    alpha={a_val:.1f}: {stat['str']}")
            
    diff_against_tex(t1, t2)
