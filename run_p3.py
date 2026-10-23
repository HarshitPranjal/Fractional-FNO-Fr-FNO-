#!/usr/bin/env python3
"""
Priority 3: DeepONet Sanity & Exploration
3a: Tuned DeepONet on S2 alpha=0.8: print final TRAIN relative L2 and TEST relative L2.
3b: Try 4 changes one at a time on S2 alpha=0.8 (same tuning grid: lr in [3e-4, 1e-3, 3e-3], wd in [0.0, 1e-4], seed 42):
    (i) input normalisation of the branch input
    (ii) trunk Fourier features of x (sin, cos for k=1..16)
    (iii) latent width 256 with parameter count kept near 287k
    (iv) 300 epochs
Report which lowers validation error. Run the best variant for 5 seeds on S2 alpha=0.8 and S1 alpha=1.2 (100 epochs).
Append all new runs to results/review_records.jsonl with stage="final_p3".
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch
import pandas as pd
import scipy.stats as st

from fr_fno_master import get_or_create_split, DATA_DIR, device

RECORDS_FILE = "results/review_records.jsonl"

def append_review_record(rec, stage="final_p3"):
    formatted = {
        "system": rec["system"],
        "alpha_true": float(rec["alpha_true"]),
        "arm": rec["arm"],
        "seed": int(rec["seed"]),
        "n_train": int(rec.get("n_train", 500)),
        "epochs": int(rec["epochs"]),
        "batch_size": int(rec.get("batch_size", 125)),
        "optimizer": "AdamW",
        "lr": float(rec["lr"]),
        "weight_decay": float(rec["weight_decay"]),
        "test_rel_l2": float(rec["test_rel_l2"]),
        "test_mse": float(rec["test_mse"]),
        "test_smape": float(rec["test_smape"]),
        "test_mape_eps": float(rec["test_mape_eps"]),
        "test_mape_clamped": float(rec["test_mape_clamped"]),
        "params": int(rec["params"]),
        "train_seconds": float(rec["train_seconds"]),
        "infer_ms_per_sample": float(rec["infer_ms_per_sample"]),
        "stage": stage,
    }
    with open(RECORDS_FILE, "a") as f:
        f.write(json.dumps(formatted) + "\n")
    return formatted

# =============================================================================
# Architecture Variants of DeepONet
# =============================================================================

# Baseline DeepONet
class BaselineDeepONet(torch.nn.Module):
    def __init__(self, sensors=256, latent=128, branch_h1=310, branch_h2=397, trunk_h=128):
        super().__init__()
        self.arm = "DeepONet"
        self.latent = latent
        self.branch = torch.nn.Sequential(
            torch.nn.Linear(sensors, branch_h1),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h1, branch_h2),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h2, latent),
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(1, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, latent),
        )
        self.b0 = torch.nn.Parameter(torch.zeros(1))
        grid = torch.linspace(0, 2.0 * math.pi, sensors).unsqueeze(1)
        self.register_buffer("grid", grid)

    def forward(self, u0):
        if u0.dim() == 3:
            u0 = u0.squeeze(1)
        b = self.branch(u0)
        t = self.trunk(self.grid)
        out = torch.einsum("bl,sl->bs", b, t) + self.b0
        return out.unsqueeze(1)

# Variant (i): Input normalization on branch input
class DeepONetInputNorm(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0, sensors=256, latent=128, branch_h1=310, branch_h2=397, trunk_h=128):
        super().__init__()
        self.arm = "DeepONet_InputNorm"
        self.latent = latent
        self.register_buffer("u_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("u_std", torch.tensor(std, dtype=torch.float32))
        self.branch = torch.nn.Sequential(
            torch.nn.Linear(sensors, branch_h1),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h1, branch_h2),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h2, latent),
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(1, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, latent),
        )
        self.b0 = torch.nn.Parameter(torch.zeros(1))
        grid = torch.linspace(0, 2.0 * math.pi, sensors).unsqueeze(1)
        self.register_buffer("grid", grid)

    def forward(self, u0):
        if u0.dim() == 3:
            u0 = u0.squeeze(1)
        u0_norm = (u0 - self.u_mean) / (self.u_std + 1.0e-7)
        b = self.branch(u0_norm)
        t = self.trunk(self.grid)
        out = torch.einsum("bl,sl->bs", b, t) + self.b0
        return out.unsqueeze(1)

# Variant (ii): Trunk Fourier Features (sin, cos for k=1..16 -> 32 dims)
class DeepONetFourierTrunk(torch.nn.Module):
    def __init__(self, sensors=256, latent=128, branch_h1=310, branch_h2=397, trunk_h=128):
        super().__init__()
        self.arm = "DeepONet_FourierTrunk"
        self.latent = latent
        self.branch = torch.nn.Sequential(
            torch.nn.Linear(sensors, branch_h1),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h1, branch_h2),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h2, latent),
        )
        # 16 Fourier modes -> 32 dims
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(32, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, latent),
        )
        self.b0 = torch.nn.Parameter(torch.zeros(1))
        grid = torch.linspace(0, 2.0 * math.pi, sensors).unsqueeze(1)  # (256, 1)
        k_vals = torch.arange(1, 17, dtype=torch.float32).unsqueeze(0)  # (1, 16)
        angles = grid * k_vals  # (256, 16)
        fourier_grid = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (256, 32)
        self.register_buffer("fourier_grid", fourier_grid)

    def forward(self, u0):
        if u0.dim() == 3:
            u0 = u0.squeeze(1)
        b = self.branch(u0)
        t = self.trunk(self.fourier_grid)
        out = torch.einsum("bl,sl->bs", b, t) + self.b0
        return out.unsqueeze(1)

# Variant (iii): Latent width 256, param-matched near 287k
class DeepONetLatent256(torch.nn.Module):
    def __init__(self, sensors=256, latent=256, branch_h1=290, branch_h2=300, trunk_h=128):
        super().__init__()
        self.arm = "DeepONet_Latent256"
        self.latent = latent
        self.branch = torch.nn.Sequential(
            torch.nn.Linear(sensors, branch_h1),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h1, branch_h2),
            torch.nn.GELU(),
            torch.nn.Linear(branch_h2, latent),
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(1, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, trunk_h),
            torch.nn.GELU(),
            torch.nn.Linear(trunk_h, latent),
        )
        self.b0 = torch.nn.Parameter(torch.zeros(1))
        grid = torch.linspace(0, 2.0 * math.pi, sensors).unsqueeze(1)
        self.register_buffer("grid", grid)

    def forward(self, u0):
        if u0.dim() == 3:
            u0 = u0.squeeze(1)
        b = self.branch(u0)
        t = self.trunk(self.grid)
        out = torch.einsum("bl,sl->bs", b, t) + self.b0
        return out.unsqueeze(1)

# =============================================================================
# Helper: Train and Eval DeepONet variant
# =============================================================================

def train_eval_deeponet(model, x_tr, y_tr, x_va, y_va, x_te, y_te, lr, wd, epochs=100, batch_size=125):
    dev = x_tr.device
    model = model.to(dev)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-5)
    criterion = torch.nn.MSELoss()
    
    n_train = len(x_tr)
    n_batches = (n_train + batch_size - 1) // batch_size
    
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_tr[b_idx])
            loss = criterion(pred, y_tr[b_idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
    wall_sec = time.time() - t0
    
    model.eval()
    with torch.no_grad():
        # Train relative L2
        pred_tr = model(x_tr)
        train_rel_l2 = (torch.norm(pred_tr - y_tr) / (torch.norm(y_tr) + 1.0e-10)).item()
        
        # Val relative L2
        pred_va = model(x_va)
        val_rel_l2 = (torch.norm(pred_va - y_va) / (torch.norm(y_va) + 1.0e-10)).item()
        
        # Test metrics
        for _ in range(3):
            _ = model(x_te)
        if dev.type == "mps":
            torch.mps.synchronize()
        t_inf0 = time.perf_counter()
        pred_te = model(x_te)
        if dev.type == "mps":
            torch.mps.synchronize()
        t_inf1 = time.perf_counter()
        infer_ms_per_sample = float(((t_inf1 - t_inf0) / len(x_te)) * 1000.0)
        
        test_rel_l2 = (torch.norm(pred_te - y_te) / (torch.norm(y_te) + 1.0e-10)).item()
        test_mse = criterion(pred_te, y_te).item()
        test_smape = (100.0 * torch.mean(2.0 * torch.abs(pred_te - y_te) / (torch.abs(pred_te) + torch.abs(y_te) + 1.0e-6))).item()
        eps_mape = 0.01 * (torch.std(y_te) + 1.0e-10)
        test_mape_eps = (100.0 * torch.mean(torch.abs(pred_te - y_te) / (torch.abs(y_te) + eps_mape))).item()
        mask_clamp = torch.abs(y_te) >= 1.0e-4
        test_mape_clamped = float(torch.mean(torch.abs(pred_te[mask_clamp] - y_te[mask_clamp]) / torch.abs(y_te[mask_clamp])).item()) if mask_clamp.sum() > 0 else 0.0

    return {
        "train_rel_l2": train_rel_l2,
        "val_rel_l2": val_rel_l2,
        "test_rel_l2": test_rel_l2,
        "test_mse": test_mse,
        "test_smape": test_smape,
        "test_mape_eps": test_mape_eps,
        "test_mape_clamped": test_mape_clamped,
        "params": params,
        "train_seconds": wall_sec,
        "infer_ms_per_sample": infer_ms_per_sample,
    }


def main():
    print("=" * 80)
    print("PRIORITY 3: DeepONet SANITY & ARCHITECTURAL EXPLORATION")
    print("=" * 80)
    
    # Load S2 alpha=0.8 dataset
    filepath_s2 = os.path.join(DATA_DIR, "S2_allen_cahn_alpha_0.8.pt")
    ds_s2 = torch.load(filepath_s2, weights_only=False)
    tr_idx, va_idx, te_idx = get_or_create_split(ds_s2, seed=42, train_size=500)
    
    x_tr = ds_s2["u0"][tr_idx].unsqueeze(1).to(device)
    y_tr = ds_s2["uT"][tr_idx].unsqueeze(1).to(device)
    x_va = ds_s2["u0"][va_idx].unsqueeze(1).to(device)
    y_va = ds_s2["uT"][va_idx].unsqueeze(1).to(device)
    x_te = ds_s2["u0"][te_idx].unsqueeze(1).to(device)
    y_te = ds_s2["uT"][te_idx].unsqueeze(1).to(device)

    # -------------------------------------------------------------------------
    # 3a. Baseline Tuned DeepONet on S2 alpha=0.8: Train vs Test Error
    # -------------------------------------------------------------------------
    print("\n--- 3a: Baseline Tuned DeepONet on S2 alpha=0.8 (lr=3e-3, wd=0.0, 100 epochs, seed 42) ---")
    torch.manual_seed(42)
    m_base = BaselineDeepONet()
    res_3a = train_eval_deeponet(m_base, x_tr, y_tr, x_va, y_va, x_te, y_te, lr=3.0e-3, wd=0.0, epochs=100)
    
    print(f"[*] Baseline DeepONet Parameters: {res_3a['params']:,}")
    print(f"[*] Final TRAIN relative L2:       {res_3a['train_rel_l2']:.4f}")
    print(f"[*] Final VAL relative L2:         {res_3a['val_rel_l2']:.4f}")
    print(f"[*] Final TEST relative L2:        {res_3a['test_rel_l2']:.4f}")
    
    if res_3a["train_rel_l2"] > 0.2:
        print(f"[!] DIAGNOSIS: TRAIN relative L2 is {res_3a['train_rel_l2']:.4f} > 0.2. The model UNDERFITS!")
    else:
        print(f"[!] DIAGNOSIS: TRAIN relative L2 is {res_3a['train_rel_l2']:.4f} <= 0.2. The model does NOT underfit on training.")

    # -------------------------------------------------------------------------
    # 3b. Try 4 changes one at a time across the tuning grid on S2 alpha=0.8 (seed 42)
    # Grid: lr in [3e-4, 1e-3, 3e-3], wd in [0.0, 1e-4]
    # -------------------------------------------------------------------------
    print("\n--- 3b: Exploration of 4 Architectural Variants on Tuning Grid (Seed 42) ---")
    grid_lrs = [3.0e-4, 1.0e-3, 3.0e-3]
    grid_wds = [0.0, 1.0e-4]
    
    u_mean = float(x_tr.mean().item())
    u_std = float(x_tr.std().item())
    
    variants = {
        "(i) Input Normalization": lambda: DeepONetInputNorm(mean=u_mean, std=u_std),
        "(ii) Trunk Fourier Features": lambda: DeepONetFourierTrunk(),
        "(iii) Latent Width 256": lambda: DeepONetLatent256(),
        "(iv) 300 Epochs (Baseline)": lambda: BaselineDeepONet(),
    }
    
    variant_results = {}
    
    for v_name, v_builder in variants.items():
        print(f"\nEvaluating Variant: {v_name}")
        best_val = float("inf")
        best_cfg = None
        best_res = None
        
        epochs_v = 300 if "300 Epochs" in v_name else 100
        
        for lr in grid_lrs:
            for wd in grid_wds:
                torch.manual_seed(42)
                m = v_builder()
                res = train_eval_deeponet(m, x_tr, y_tr, x_va, y_va, x_te, y_te, lr=lr, wd=wd, epochs=epochs_v)
                print(f"  lr={lr:.1e}, wd={wd:.1e} -> Val Rel L2 = {res['val_rel_l2']:.4f}, Test Rel L2 = {res['test_rel_l2']:.4f}, Train Rel L2 = {res['train_rel_l2']:.4f}")
                if res["val_rel_l2"] < best_val:
                    best_val = res["val_rel_l2"]
                    best_cfg = (lr, wd)
                    best_res = res
        
        variant_results[v_name] = {
            "best_cfg": best_cfg,
            "best_val": best_val,
            "best_res": best_res,
            "epochs": epochs_v,
        }
        print(f"-> Best for {v_name}: lr={best_cfg[0]:.1e}, wd={best_cfg[1]:.1e} | Val = {best_val:.4f} | Test = {best_res['test_rel_l2']:.4f} | Train = {best_res['train_rel_l2']:.4f} | Params = {best_res['params']:,}")

    # Summary table of 3b exploration
    print("\n" + "=" * 80)
    print("SUMMARY OF ARCHITECTURAL EXPLORATION (S2 alpha=0.8, Seed 42)")
    print("=" * 80)
    print(f"{'Variant':<30} | {'Best (lr, wd)':<18} | {'Val Rel L2':<10} | {'Test Rel L2':<11} | {'Train Rel L2':<12} | {'Params':<8}")
    print("-" * 105)
    print(f"{'Original Tuned Baseline':<30} | {'(3.0e-3, 0.0e+00)':<18} | {res_3a['val_rel_l2']:<10.4f} | {res_3a['test_rel_l2']:<11.4f} | {res_3a['train_rel_l2']:<12.4f} | {res_3a['params']:<8,}")
    for v_name, v_data in variant_results.items():
        cfg_str = f"({v_data['best_cfg'][0]:.1e}, {v_data['best_cfg'][1]:.1e})"
        print(f"{v_name:<30} | {cfg_str:<18} | {v_data['best_val']:<10.4f} | {v_data['best_res']['test_rel_l2']:<11.4f} | {v_data['best_res']['train_rel_l2']:<12.4f} | {v_data['best_res']['params']:<8,}")

    # Determine best variant
    best_v_name = min(variant_results, key=lambda k: variant_results[k]["best_val"])
    best_v_data = variant_results[best_v_name]
    baseline_val = res_3a["val_rel_l2"]
    
    print("\n" + "=" * 80)
    if best_v_data["best_val"] < baseline_val:
        print(f"[+] BEST VARIANT: {best_v_name} lowered validation error from {baseline_val:.4f} to {best_v_data['best_val']:.4f}!")
    else:
        print(f"[-] None of the variants beat baseline validation error {baseline_val:.4f}.")
    
    if all(v_data["best_res"]["test_rel_l2"] > 0.2 for v_data in variant_results.values()):
        print("[!] Note: All variants remain strictly above 0.2 relative L2 error.")
    
    # Run the best variant (at 100 epochs as requested: "run the best variant for 5 seeds on S2 alpha=0.8 and S1 alpha=1.2 at 100 epochs. Report final params. If nothing brings it below 0.2, say so and keep the existing numbers.")
    # Notice: if (iv) was best, prompt explicitly states "at 100 epochs", so if (iv) was only best because of 300 epochs, we evaluate the best architectural variant among i, ii, iii at 100 epochs, or best overall at 100 epochs.
    arch_variants = {k: v for k, v in variant_results.items() if k != "(iv) 300 Epochs (Baseline)"}
    best_arch_name = min(arch_variants, key=lambda k: arch_variants[k]["best_val"])
    print(f"\nRunning Best Architectural Variant at 100 epochs across 5 seeds: {best_arch_name}")
    best_arch_builder = variants[best_arch_name]
    best_arch_lr, best_arch_wd = arch_variants[best_arch_name]["best_cfg"]

    # 5 seeds on S2 alpha=0.8 and S1 alpha=1.2
    eval_cells = [
        ("S2_allen_cahn", 0.8),
        ("S1_burgers", 1.2),
    ]
    eval_seeds = [42, 123, 999, 2024, 2025]
    
    p3_5seed_records = []
    
    for sys_name, a_val in eval_cells:
        filepath = os.path.join(DATA_DIR, f"{sys_name}_alpha_{a_val:.1f}.pt")
        ds = torch.load(filepath, weights_only=False)
        print(f"\n--- 5-seed evaluation on {sys_name} alpha={a_val} ({best_arch_name}) ---")
        
        for s in eval_seeds:
            tr_i, va_i, te_i = get_or_create_split(ds, seed=s, train_size=500)
            x_tr_s = ds["u0"][tr_i].unsqueeze(1).to(device)
            y_tr_s = ds["uT"][tr_i].unsqueeze(1).to(device)
            x_va_s = ds["u0"][va_i].unsqueeze(1).to(device)
            y_va_s = ds["uT"][va_i].unsqueeze(1).to(device)
            x_te_s = ds["u0"][te_i].unsqueeze(1).to(device)
            y_te_s = ds["uT"][te_i].unsqueeze(1).to(device)
            
            torch.manual_seed(s)
            m = best_arch_builder()
            res = train_eval_deeponet(m, x_tr_s, y_tr_s, x_va_s, y_va_s, x_te_s, y_te_s, lr=best_arch_lr, wd=best_arch_wd, epochs=100)
            
            rec = {
                "system": sys_name,
                "alpha_true": a_val,
                "arm": f"DeepONet_{best_arch_name.split()[1]}",
                "seed": s,
                "n_train": 500,
                "epochs": 100,
                "batch_size": 125,
                "lr": best_arch_lr,
                "weight_decay": best_arch_wd,
                "test_rel_l2": res["test_rel_l2"],
                "test_mse": res["test_mse"],
                "test_smape": res["test_smape"],
                "test_mape_eps": res["test_mape_eps"],
                "test_mape_clamped": res["test_mape_clamped"],
                "params": res["params"],
                "train_seconds": res["train_seconds"],
                "infer_ms_per_sample": res["infer_ms_per_sample"],
            }
            formatted = append_review_record(rec, stage="final_p3")
            p3_5seed_records.append(formatted)
            print(f"  Seed {s:<4}: Test Rel L2 = {res['test_rel_l2']:.4f}, Train Rel L2 = {res['train_rel_l2']:.4f}, Val Rel L2 = {res['val_rel_l2']:.4f}")

    df_p3_5seed = pd.DataFrame(p3_5seed_records)
    df_p3_5seed.to_csv("results/priority3_deeponet_5seed.csv", index=False)
    
    # Summary of 5 seeds
    print("\n" + "=" * 80)
    print(f"5-SEED STATS FOR BEST VARIANT: {best_arch_name}")
    print("=" * 80)
    for (sys_name, a_val), grp in df_p3_5seed.groupby(["system", "alpha_true"]):
        vals = grp["test_rel_l2"]
        print(f"Cell {sys_name} alpha={a_val}: Mean = {vals.mean():.4f} +/- {vals.std(ddof=1):.4f} (95% CI: [{vals.mean() - 2.776*vals.std(ddof=1)/np.sqrt(5):.4f}, {vals.mean() + 2.776*vals.std(ddof=1)/np.sqrt(5):.4f}])")
    
    print("\n[+] Priority 3 Complete!")

if __name__ == "__main__":
    main()
