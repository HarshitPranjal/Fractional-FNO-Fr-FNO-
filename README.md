# Adaptive Fractional Fourier Neural Operators for Learning Nonlocal PDE Dynamics

[![Paper](https://img.shields.io/badge/IEEE%20INDICON-2026-blue)](main.pdf)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and empirical audit suite for the paper:  
**"Adaptive Fractional Fourier Neural Operators for Learning Nonlocal PDE Dynamics"**  
*23rd IEEE India Council International Conference (INDICON 2026)* — Paper ID 1472.

---

## 📌 Overview

Fourier Neural Operators (FNOs) learn resolution-invariant solution mappings of partial differential equations (PDEs) in Fourier space. However, standard spectral layers lack an explicit parameterization of the nonlocal, power-law dissipation characteristic of fractional differential operators $(-\Delta)^{\alpha/2}$.

This repository provides:
1. **Mathematical Filter Derivation**: Formal analysis showing that heuristic rational filters $\Psi(k) = [e^{1-\alpha} + \alpha(k/M)^2]^{-1}$ possess a fixed $\mathcal{O}(k^{-2})$ quadratic roll-off regardless of $\alpha$, whereas the true **Fractional Resolvent Filter** $\Psi(k) = [1 + \beta |k/M|^\alpha]^{-1}$ strictly exhibits $\mathcal{O}(k^{-\alpha})$ power-law decay matching the fractional Laplacian.
2. **Comprehensive Operator Benchmark**: Rigorous, controlled evaluation across:
   - **FNO Variants**: Vanilla FNO ($A_0$), Parameter-Matched FNO ($A_1$), Gaussian Smoothing ($A_2$), Fixed fr-FNO ($A_3$), Adaptive fr-FNO ($A_5$), True Dissipative fr-FNO ($A_6$), Normalized Dissipative ($A_{6j}$), Resolvent Filter ($A_{10}$), and Spectral Multiplier ($A_{9a}$).
   - **Alternative Operator Baselines**: Parameter-matched **1D DeepONet** (branch-trunk architecture) and **1D U-Net** (3-level CNN), strictly tuned under identical budgets (~287k parameters).
3. **Statistical Rigor & Paired CIs**: Multi-seed evaluations ($n=5$) reporting 95% Student-$t$ paired-difference confidence intervals $[\bar{\Delta} \pm t_{.975}\frac{s}{\sqrt{n}}]$ vs. $A_0$, confirming statistically significant superiority on fractional phase-field systems ($p < 0.05$) and honest parity on advective shock systems.
4. **Multi-Metric Suite (Addressing Zero-Crossings)**: Evaluation across Relative $L^2$, Mean Squared Error (MSE), Symmetric MAPE (sMAPE), $\epsilon$-regularized MAPE ($\text{MAPE}_\epsilon$ with $\epsilon = 0.01 \cdot \text{std}(u)$), and clamped raw MAPE, explaining the divergence of raw percentage error near nodal zero-crossings.
5. **Long-Horizon Scaling & High-Order Scope**: 300-epoch extended convergence on Allen-Cahn and evaluation on high fractional orders ($\alpha \in \{1.6, 1.9\}$).

---

## 🔬 Key Empirical Findings

- **Operator Superiority**: FNO outperforms DeepONet by $11\times$ to $42\times$ ($p < 10^{-6}$) and U-Net by $3\times$ to $4\times$ ($p < 10^{-3}$) under identical ~287k parameter budgets, demonstrating that non-local fractional Laplacians require global spectral receptive fields.
- **Physical Dissipative Advantage**: On Fractional Allen-Cahn ($\alpha=0.8$), the true dissipative filter ($A_6$) reduces relative $L^2$ error by **45.2%** over vanilla FNO ($0.0070$ vs. $0.0128$, paired difference 95% CI $[-0.0106, -0.0010]$, $p = 0.0292^*$).
- **Advective Wave Parity**: On Fractional Burgers ($\alpha=1.2$), where nonlinear hyperbolic wave steepening counterbalances dissipation, vanilla FNO ($0.0389$) achieves parity with the resolvent filter ($0.0392$) and dissipative filter ($0.0399$), while rigid heuristic filters ($A_3, A_5$) over-smooth shock fronts.
- **Minimal Compute Overhead**: Adding fractional filters incurs under ~10% compute overhead ($0.117 \to 0.125$ s/epoch), well within hardware timing noise.
- **Singularity of Raw MAPE**: Across test sets, $0.02\% - 0.22\%$ of points have $|u| < 10^{-3}$ with minima reaching $2\times 10^{-6}$. Dividing by near-zero values explodes unregularized raw MAPE to $>100\%$, while $\text{sMAPE}$ and $\text{MAPE}_\epsilon$ provide stable, meaningful metrics.

---

## 📊 Benchmark Results

### 1. Primary Test Relative $L^2$ Error ($N_{\text{tr}}=500$, 100 Epochs, Mean $\pm$ Std over 5 Seeds)
*All models tuned on identical AdamW grid ($lr \in \{3\times 10^{-4}, 1\times 10^{-3}, 3\times 10^{-3}\}$, $wd \in \{0, 10^{-4}\}$, batch size 125).*

| Model Architecture | Params | S1 (Burgers) $\alpha=1.2$ | S2 (Allen-Cahn) $\alpha=0.8$ | S2 (Allen-Cahn) $\alpha=1.2$ |
| :--- | :---: | :---: | :---: | :---: |
| **$A_0$ (Vanilla FNO)** | 287,361 | $0.0389 \pm 0.0063$ | $0.0128 \pm 0.0038$ | $0.0151 \pm 0.0012$ |
| **$A_3$ (Fixed fr-FNO)** | 287,361 | $0.0575 \pm 0.0075$ | $\mathbf{0.0078 \pm 0.0005}$ | $0.0161 \pm 0.0005$ |
| **$A_5$ (Adaptive fr-FNO)** | 287,617 | $0.0554 \pm 0.0100$ | $0.0079 \pm 0.0008$ | $\mathbf{0.0148 \pm 0.0004}$ |
| **$A_6$ (True Dissipative)** | 287,873 | $0.0399 \pm 0.0048$ | $\mathbf{0.0070 \pm 0.0011}$ | $0.0164 \pm 0.0007$ |
| **$A_{6j}$ (Norm. Dissipative)** | 287,873 | $0.0392 \pm 0.0060$ | $0.0126 \pm 0.0042$ | $0.0151 \pm 0.0016$ |
| **$A_{10}$ (Resolvent Filter)** | 287,873 | $0.0392 \pm 0.0060$ | $0.0126 \pm 0.0042$ | $0.0151 \pm 0.0016$ |
| **$A_{9a}$ (Init Multiplier)** | 287,361 | $\mathbf{0.0382 \pm 0.0060}$ | $0.0127 \pm 0.0034$ | $0.0151 \pm 0.0010$ |
| \midrule
| **U-Net 1D Baseline** | 287,360 | $0.1674 \pm 0.0043$ | $0.0447 \pm 0.0054$ | $0.0443 \pm 0.0068$ |
| **DeepONet 1D Baseline** | 287,362 | $0.4518 \pm 0.0090$ | $0.5375 \pm 0.0121$ | $0.5397 \pm 0.0214$ |

### 2. Paired-Difference Confidence Intervals ($A - A_0$)
- **S2 Allen-Cahn ($\alpha=0.8$)**:
  - $A_6 - A_0$: mean diff $\mathbf{-0.00577}$, 95% CI $\mathbf{[-0.01059, -0.00095]}$ $\implies$ **Statistically significantly better than $A_0$ ($p < 0.05$)**.
  - $A_3 - A_0$: mean diff $\mathbf{-0.00494}$, 95% CI $\mathbf{[-0.00952, -0.00035]}$ $\implies$ **Statistically significantly better than $A_0$ ($p < 0.05$)**.
- **S1 Burgers ($\alpha=1.2$)**:
  - $A_{10}, A_{6j}, A_6, A_{9a}$: CIs all span 0 ($\bar{\Delta} \in [-0.0007, +0.0010]$) $\implies$ **Exact parity with $A_0$**.
  - DeepONet ($+0.413$) and U-Net ($+0.129$): CIs strictly positive $\implies$ **Significantly worse than $A_0$**.

### 3. Extended 300-Epoch Scaling (S2 Allen-Cahn $\alpha=1.2$)
- $A_{10}$ (Resolvent): $\mathbf{0.01066 \pm 0.00009}$ (CI $\pm 0.00022$)
- $A_0$ (Vanilla FNO): $\mathbf{0.01068 \pm 0.00030}$ (CI $\pm 0.00076$)
- $A_6$ (True Dissipative): $0.01152 \pm 0.00006$
- U-Net 1D: $0.01997 \pm 0.00088$
- DeepONet 1D: $0.43203 \pm 0.00550$

---

## 📁 Repository Structure

```
├── README.md                                    # Documentation and user guide
├── LICENSE                                      # MIT License
├── requirements.txt                             # Python library dependencies
├── .gitignore                                   # Git ignore rules for build and temp files
├── config_dump.txt                              # Exhaustive technical configuration dump
├── fr_fno_master.py                             # Core models, solvers, training, and probing
├── audit_suite.py                               # Comprehensive replication and figure generation
├── run_final_pipeline.py                        # Orchestrates tuning parity and 5-seed confirmation
├── run_tier_c1.py                               # High-order fractional evaluation (alpha=1.6, 1.9)
├── make_tables.py                               # Script for LaTeX table generation
├── main.tex                                     # IEEE conference manuscript source (5 pages)
├── main.pdf                                     # Compiled publication-ready PDF
├── References.bib                               # Complete bibliography file
├── IEEEtran.cls / .bst                          # IEEEtran document class and BibTeX style
├── data/                                        # 16 benchmark datasets (S1-S4 across 4 fractional orders)
│   ├── S1_burgers_alpha_*.pt
│   ├── S2_allen_cahn_alpha_*.pt
│   ├── S3_diffusion_alpha_*.pt
│   └── S4_fisher_kpp_alpha_*.pt
├── figures/                                     # Publication-ready vector figures (PDF)
│   ├── fig_curves.pdf                           # Loss curves (mean ± std over 5 seeds)
│   ├── fig_qualitative.pdf                      # Predicted profiles and pointwise error
│   ├── fig_spectral_decay_comparison.pdf        # Theoretical roll-off comparison
│   └── alpha_eff.pdf                            # Symbol probe correlation
├── tables/                                      # Exported LaTeX table snippets
│   ├── tab_core.tex                             # Core benchmark table
│   ├── tab_core_ddof1.tex                       # Core benchmark table with sample std (ddof=1)
│   ├── tab_selected_hparams.tex                 # Selected hyperparameters & validation errors
│   ├── tab_secondary.tex                        # Secondary error metrics (sMAPE, MAPE_eps, clamped)
│   ├── tab_p1_300epoch.tex                      # 300-epoch extended evaluation (seeds 42, 123, 999)
│   ├── tab_p4_n900.tex                          # Full-data (N_tr=900) evaluation
│   ├── tab_ablation.tex                         # Ablation table
│   └── tab_reviewer_metrics.tex                 # Reviewer metric comparison
├── run_p1.py                                    # Priority 1: 300-epoch extension suite
├── process_p2.py                                # Priority 2: Secondary metrics, Holm correction, Wilcoxon
├── extract_2d_params.py                         # Priority 2d: Layer-wise parameter movement measurement
├── run_p3.py                                    # Priority 3: DeepONet underfitting & architectural variants
├── run_p4.py                                    # Priority 4: N_tr=900 full training index benchmark
└── results/                                     # Complete empirical records and CSV summaries
    ├── review_records.jsonl                     # Full append-only log of 439 experiment runs
    ├── priority1_300epoch_stats.csv             # P1 300-epoch summary statistics & paired CIs
    ├── priority2d_params.csv                    # P2d final alpha/beta parameter values per arm
    ├── priority3_deeponet_5seed.csv             # P3 5-seed confirmation of Fourier-trunk DeepONet
    ├── priority4_n900_stats.csv                 # P4 N_tr=900 summary statistics & paired CIs
    ├── paired_difference_confidence_intervals.csv # 95% paired Student-t CIs vs A0
    ├── final_5seed_confidence_intervals.csv     # Individual 95% CIs per model
    ├── extended_300epoch_confidence_intervals.csv # 300-epoch evaluation
    ├── secondary_metrics_summary.csv            # MSE, sMAPE, MAPE_eps, clamped MAPE
    ├── compute_cost_benchmark.csv               # Parameters, throughput, latency
    ├── tuning_parity_chosen.csv                 # Optimal hyperparameters per cell
    └── tier_c1_summary.csv                      # High-order alpha=1.6, 1.9 summary
```

---

## 🔬 Final Reviewer Exploration (Priorities 1–4)

All exploratory runs use the strict frozen protocol (`batch_size=125` for $N_{\text{tr}} \ge 500$, `AdamW`, $\eta_{\min}=10^{-5}$, MPS device), logged with `"stage": "final_p1" / "final_p3" / "final_p4"`.

### Priority 1: 300-Epoch Scaling (30 Runs)
- **Cell S2 Allen-Cahn ($\alpha=0.8$):** $A_6$ achieves $0.0035 \pm 0.0002$ vs. $A_0 = 0.0040 \pm 0.0003$.
- **Cell S1 Burgers ($\alpha=1.2$):** $A_6$ achieves $0.0185 \pm 0.0007$ vs. $A_0 = 0.0269 \pm 0.0031$ ($p = 0.035^*$).

### Priority 2: Statistical Rigor & Metric Clarification
- **Multiple Comparisons:** DeepONet and U-Net across all cells remain strictly inferior under conservative Holm-Bonferroni correction ($p_{\text{adj}} \le 1.9\times 10^{-3}$).
- **Exact Wilcoxon Signed-Rank:** On S2 $\alpha=0.8$ ($n=5$), filter arms ($A_3, A_5, A_6$) beat $A_0$ on $100\%$ of seeds ($W=0.0$). The exact two-sided $p = 0.0625$ represents the theoretical floor for $n=5$.
- **Sample Std Convention:** All core tables regenerated with unbiased sample standard deviation ($ddof=1$, [`tables/tab_core_ddof1.tex`](tables/tab_core_ddof1.tex)).
- **Filter Order Parameter Movement:** $A_5$ consistently converges to a numerical filter scale ($\sim 0.84$ on Burgers, $\sim 1.01$ on AC), while $A_{6j}$ and $A_{10}$ remain near initialization ($\beta \approx 0.3$).

### Priority 3: DeepONet Inductive Bias & Sanity Audit
- **Underfitting Diagnosis:** Tuned baseline DeepONet on S2 $\alpha=0.8$ has train relative $L^2 = 0.5700 > 0.20$ and test relative $L^2 = 0.5606$, proving inherent underfitting rather than overfitting.
- **Architectural Exploration:** Adding trunk Fourier features ($k=1..16$) reduces test error to $0.2486 \pm 0.0026$ (Burgers) and $0.2546 \pm 0.0039$ (AC). However, no variant reaches $<0.20$, confirming that global spectral convolution is indispensable.

### Priority 4: Large-Data Benchmark ($N_{\text{tr}}=900$, 40 Runs)
- Evaluating the entire training partition ($N_{\text{tr}}=900$, 5 seeds) maintains the hierarchy: $A_6$ achieves $0.0047 \pm 0.0008$ on S2 AC $\alpha=0.8$ and $0.0220 \pm 0.0005$ on S1 Burgers $\alpha=1.2$, while $A_0$ achieves $0.0055 \pm 0.0013$ and $0.0233 \pm 0.0027$.


---

## 🚀 Quick Start & Replication

### 1. Installation

```bash
git clone https://github.com/HarshitPranjal/Fractional-FNO-Fr-FNO-.git
cd Fractional-FNO-Fr-FNO-
pip install -r requirements.txt
```

### 2. Run the Full Revision Audit Suite

To re-run the entire tuning parity, 5-seed confirmation, and figure generation pipeline:
```bash
python audit_suite.py
```
This generates `figures/fig_curves.pdf`, `figures/fig_qualitative.pdf`, and logs all records to `results/review_records.jsonl`.

### 3. Evaluate Individual Operators with `fr_fno_master.py`

Train Vanilla FNO ($A_0$) on Space-Fractional Burgers ($\alpha=1.2$):
```bash
python fr_fno_master.py --system S1_burgers --alpha 1.2 --arm A0 --train_size 500 --epochs 100 --seed 42
```

Train True Fractional Dissipative Filter ($A_6$):
```bash
python fr_fno_master.py --system S2_allen_cahn --alpha 0.8 --arm A6 --train_size 500 --epochs 100 --seed 42
```

Train Resolvent Filter ($A_{10}$):
```bash
python fr_fno_master.py --system S2_allen_cahn --alpha 1.2 --arm A10 --train_size 500 --epochs 100 --seed 42
```

Train DeepONet 1D Baseline:
```bash
python fr_fno_master.py --system S1_burgers --alpha 1.2 --arm DeepONet --train_size 500 --epochs 100 --seed 42
```

Train U-Net 1D Baseline:
```bash
python fr_fno_master.py --system S1_burgers --alpha 1.2 --arm UNet --train_size 500 --epochs 100 --seed 42
```

### 4. Compile Manuscript

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## 📜 Citation

```bibtex
@inproceedings{pranjal2026adaptive,
  title={Adaptive Fractional Fourier Neural Operators for Learning Nonlocal PDE Dynamics},
  author={Pranjal, Harshit},
  booktitle={23rd IEEE India Council International Conference (INDICON)},
  year={2026},
  organization={IEEE}
}
```

---

## 📄 License

This repository is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
