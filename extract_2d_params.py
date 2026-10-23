#!/usr/bin/env python3
"""
Priority 2d: Parameter movement for A5, A6, A6j, A10 at tuned settings for seed 42
on S2 alpha=0.8 and S1 alpha=1.2 (100 epochs, n_train=500, batch_size=125).
Report final alpha and beta (mean +- std over channels and blocks).
Check if A6j and A10 stay near their init and confirm the two arms are not aliased in code.
"""

import os
import torch
import numpy as np
import pandas as pd
from fr_fno_master import UnifiedFNO1d, get_or_create_split, build_optimizer_and_groups, DATA_DIR, device

cells = [
    ("S2_allen_cahn", 0.8),
    ("S1_burgers", 1.2),
]
arms = ["A5", "A6", "A6j", "A10"]
seed = 42
epochs = 100
batch_size = 125
lr = 3.0e-3
wd = 0.0

results = []

for sys_name, a_val in cells:
    filepath = os.path.join(DATA_DIR, f"{sys_name}_alpha_{a_val:.1f}.pt")
    ds = torch.load(filepath, weights_only=False)
    tr_idx, val_idx, te_idx = get_or_create_split(ds, seed=seed, train_size=500)
    
    x_tr = ds["u0"][tr_idx].unsqueeze(1).to(device)
    y_tr = ds["uT"][tr_idx].unsqueeze(1).to(device)
    x_te = ds["u0"][te_idx].unsqueeze(1).to(device)
    y_te = ds["uT"][te_idx].unsqueeze(1).to(device)
    
    for arm in arms:
        torch.manual_seed(seed)
        model = UnifiedFNO1d(arm=arm, width=64, modes=16, num_blocks=4).to(device)
        optimizer = build_optimizer_and_groups(model, base_lr=lr, alpha_lr_mult=10.0, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-5)
        criterion = torch.nn.MSELoss()
        
        n_train = len(x_tr)
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n_train, device=device)
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = model(x_tr[idx])
                loss = criterion(pred, y_tr[idx])
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            pred_te = model(x_te)
            rel_l2 = (torch.norm(pred_te - y_te) / (torch.norm(y_te) + 1.0e-10)).item()
            
            # Extract alpha across blocks & channels
            alphas = []
            betas = []
            for blk in model.spectral_blocks:
                if blk.filter is not None:
                    a = blk.filter.get_alpha()
                    if a is not None:
                        alphas.append(a.detach().cpu())
                    c = blk.filter.get_c()
                    if c is not None:
                        betas.append(c.detach().cpu())
                        
            all_a = torch.cat([a.reshape(-1) for a in alphas]) if len(alphas) > 0 else None
            all_b = torch.cat([b.reshape(-1) for b in betas]) if len(betas) > 0 else None
            
            a_mean = float(all_a.mean().item()) if all_a is not None else np.nan
            a_std = float(all_a.std(unbiased=True).item()) if all_a is not None else np.nan
            b_mean = float(all_b.mean().item()) if all_b is not None else np.nan
            b_std = float(all_b.std(unbiased=True).item()) if all_b is not None else np.nan
            
            results.append({
                "system": sys_name,
                "alpha_true": a_val,
                "arm": arm,
                "test_rel_l2": rel_l2,
                "alpha_mean": a_mean,
                "alpha_std": a_std,
                "beta_mean": b_mean,
                "beta_std": b_std,
                "num_params_measured": len(all_a) if all_a is not None else 0,
            })
            print(f"[{sys_name} a={a_val} {arm}] Rel L2: {rel_l2:.4f} | alpha: {a_mean:.4f} +/- {a_std:.4f} | beta: {b_mean:.4f} +/- {b_std:.4f}")

df_params = pd.DataFrame(results)
df_params.to_csv("results/priority2d_params.csv", index=False)
print("\n" + "=" * 80)
print("PRIORITY 2d PARAMETER MOVEMENT RESULTS (Seed 42)")
print("=" * 80)
print(df_params.to_string())
