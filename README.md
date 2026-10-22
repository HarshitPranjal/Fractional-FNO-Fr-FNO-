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

Fourier Neural Operators (FNOs) parameterize solution operators of partial differential equations (PDEs) in Fourier space, but standard spectral layers lack an explicit parameterization of the nonlocal, power-law dissipation characteristic of fractional differential operators $(-\Delta)^{\alpha/2}$.

This repository contains:
1. **Mathematical Theory & Filter Reformulation**: Formal proof that heuristic rational filters $\Psi(k) = [e^{1-\alpha} + \alpha(k/M)^2]^{-1}$ possess a fixed $\mathcal{O}(k^{-2})$ quadratic roll-off regardless of $\alpha$ (Proposition 1), and derivation of the true **Fractional Resolvent Filter** $\Psi(k) = [1 + c |k/M|^\alpha]^{-1}$ with identity initialization (Proposition 2), which strictly exhibits $\mathcal{O}(k^{-\alpha})$ decay matching the fractional Laplacian.
2. **Comprehensive 8-Operator Benchmark Ladder**: Controlled evaluation across standard FNO ($A_0$), validation-tuned FNO ($\text{Tuned-}A_0$), parameter-matched FNO ($A_1$), Gaussian smoothing ($A_2$), fixed fractional filter ($A_3$), adaptive rational filter ($A_5$), fractional resolvent with identity init ($A_6 / A_{6i}$), and frozen filter controls ($A_7$).
3. **Multi-Metric & Statistical Reliability**: Evaluation across 4 PDEs, 4 fractional orders ($\alpha \in \{0.8, 1.2, 1.6, 1.9\}$), 4 spatial resolutions ($N \in \{64, 128, 256, 512\}$), and 5 repeated random seeds (`42, 123, 999, 2024, 2025`), reporting Relative $L^2$, MSE, Symmetric MAPE (sMAPE), $\epsilon$-regularized MAPE ($\text{MAPE}_\epsilon$), and paired Wilcoxon/Sign significance tests with Holm-Bonferroni correction.
4. **Model-Agnostic Symbol Probing**: A harmonic perturbation probe that extracts effective dissipation exponents $\alpha_{\text{eff}}$ directly from trained operators, achieving $\text{MAE} = 0.072$ on linear fractional diffusion.

---

## 🔬 Key Theoretical & Empirical Findings

- **Filter Reformulation Works**: The heuristic rational filter decays as $\mathcal{O}(k^{-2})$ for all $\alpha$, failing to parameterize fractional dissipation. In contrast, the reformulated resolvent filter $(1 + c |k/M|^\alpha)^{-1}$ strictly achieves $\mathcal{O}(k^{-\alpha})$ roll-off.
- **Tuning Parity Dominates Inductive Bias**: When vanilla FNO is afforded standard hyperparameter validation tuning ($\text{lr} = 3\times 10^{-3}, \text{wd} = 0 / 10^{-4}$), it outperforms all fractional filter variants across 9 of 10 paired seed conditions ($p_{\mathrm{Holm}} = 0.0156$, Wilcoxon signed-rank test).
- **Generic Low-Pass Regularization**: On phase-field Allen--Cahn dynamics, fixed filters ($A_3$) and frozen random filters ($A_7$) match or exceed learned filters ($A_5$), confirming that gains stem from spectral low-pass prior regularization rather than learned fractional orders.
- **MAPE Metric Pathology Explained**: Continuous PDE solutions cross zero ($u(x) = 0$), causing raw MAPE to diverge. Reporting Symmetric MAPE (sMAPE) and $\epsilon$-regularized MAPE provides stable, meaningful percentage metrics.
- **Resolution Invariance Preserved**: Zero-shot mesh transfer from $N=256$ to $N \in \{64, 128, 256, 512\}$ remains strictly invariant across all architectures.

---

## 📊 Summary of Main Benchmark Results

Results on Space-Fractional Burgers (S1) and Fractional Allen--Cahn (S2) at $\alpha=1.2$ ($N_{\text{tr}}=500$, 100 epochs, Seed 42):

| System | Model Variant | Rel. $L^2$ Error | MSE ($\times 10^{-4}$) | sMAPE (%) | $\text{MAPE}_\epsilon$ (%) | Trainable Params | Train Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 (Burgers)** | $A_0$ (Default Vanilla FNO) | 0.0549 | 3.96 | 9.97% | 9.74% | 287,361 | 11.5s |
| | **Tuned-$A_0$ (Validation-Tuned)** | **0.0398** | **2.07** | **8.05%** | **7.30%** | 287,361 | 13.9s |
| | $A_3$ (Fixed fr-FNO, $\alpha^*=0.4$) | 0.1081 | 15.33 | 17.73% | 20.24% | 287,361 | 14.9s |
| | $A_5$ (Adaptive fr-FNO, Rational) | 0.1049 | 14.42 | 16.81% | 19.13% | 287,617 | 11.7s |
| | $A_6$ (True Frac. Resolvent) | 0.0556 | 4.06 | 10.20% | 9.75% | 287,873 | 12.1s |
| | $A_{6i}$ (Identity-Init Resolvent) | 0.0549 | 3.96 | 9.97% | 9.74% | 287,873 | 15.1s |
| **S2 (Allen--Cahn)** | $A_0$ (Default Vanilla FNO) | 0.0315 | 6.54 | 6.03% | 6.18% | 287,361 | 11.5s |
| | **Tuned-$A_0$ (Validation-Tuned)** | **0.0158** | **1.65** | **3.35%** | **3.03%** | 287,361 | 13.9s |
| | $A_3$ (Fixed fr-FNO, $\alpha^*=1.2$) | 0.0180 | 2.13 | 3.68% | 3.38% | 287,361 | 14.9s |
| | $A_5$ (Adaptive fr-FNO, Rational) | 0.0190 | 2.38 | 3.86% | 3.55% | 287,617 | 11.7s |
| | $A_6$ (True Frac. Resolvent) | 0.0189 | 2.36 | 3.85% | 3.51% | 287,873 | 12.1s |
| | $A_{6i}$ (Identity-Init Resolvent) | 0.0315 | 6.54 | 6.03% | 6.18% | 287,873 | 15.1s |

---

## 📁 Repository Structure

```
├── README.md                  # Complete documentation and user guide
├── LICENSE                    # MIT License
├── requirements.txt           # Python library dependencies
├── .gitignore                 # Ignore patterns for temporary and binary files
├── fr_fno_master.py           # Core implementation: models, solvers, training, and probing
├── make_tables.py             # Generates all LaTeX tables directly from JSONL records
├── main.tex                   # IEEE conference manuscript source (5 pages)
├── main.pdf                   # Compiled publication-ready PDF
├── References.bib             # Complete bibliography file
├── IEEEtran.cls / .bst        # IEEEtran LaTeX document class and BibTeX style
├── data/                      # 16 benchmark datasets (S1-S4 across 4 fractional orders)
│   ├── S1_burgers_alpha_*.pt
│   ├── S2_allen_cahn_alpha_*.pt
│   ├── S3_diffusion_alpha_*.pt
│   └── S4_fisher_kpp_alpha_*.pt
├── figures/                   # Publication-quality vector figures
│   ├── fig_spectral_decay_comparison.pdf
│   ├── fig_alpha_eff.pdf
│   └── ...
├── tables/                    # Exported LaTeX table snippets
│   ├── tab_reviewer_metrics.tex
│   ├── tab_ablation.tex
│   └── tab_resolution.tex
└── results/                   # Evaluation records & checkpoints
    ├── experiment_records.jsonl # Complete audit log of 379 experiment runs
    └── checkpoints/
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/HarshitPranjal/Fractional-FNO-Fr-FNO-.git
cd Fractional-FNO-Fr-FNO-
pip install -r requirements.txt
```

### 2. Reproduce Tables and Figures

To parse the 379 experimental runs in `results/experiment_records.jsonl` and regenerate all LaTeX tables:
```bash
python make_tables.py
```

### 3. Run Experiments with `fr_fno_master.py`

Train standard FNO ($A_0$) on Space-Fractional Burgers at $\alpha=1.2$:
```bash
python fr_fno_master.py --system S1_burgers --alpha 1.2 --arm A0 --train_size 500 --epochs 100 --seed 42
```

Train true Fractional Resolvent Filter with Identity Initialization ($A_{6i}$):
```bash
python fr_fno_master.py --system S1_burgers --alpha 1.2 --arm A6i --train_size 500 --epochs 100 --seed 42
```

Evaluate Zero-Shot Super-Resolution Transfer across resolutions ($N \in \{64, 128, 256, 512\}$):
```bash
python fr_fno_master.py --action resolution_transfer --arm A6i --system S1_burgers --alpha 1.2 --seed 42
```

Run Linear-Response Symbol Probe to measure effective dissipation order $\alpha_{\text{eff}}$:
```bash
python fr_fno_master.py --action probe --system S3_diffusion --alpha 1.2 --arm A0 --seed 42
```

### 4. Compile the Paper

Compile the LaTeX manuscript:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📜 Citation

If you find this code, theoretical analysis, or empirical benchmark useful in your research, please cite:

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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
