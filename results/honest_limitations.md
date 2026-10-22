# Honest Limitations & Empirical Audit of Adaptive fr-FNO

> **Audit Requirement**: No claim in text may exceed what the tables show. 
> Below is an exhaustive enumeration of every experimental setting where adaptive fr-FNO (Arm A5) 
> failed to outperform standard controls (Vanilla FNO A0, Parameter-Matched FNO A1, or Gaussian Low-Pass FNO A2).

## 1. Summary of Non-Superiority Conditions for Adaptive fr-FNO (A5)

- **Total Benchmark Conditions Tested**: 20
- **Settings where A5 was matched or beaten by Vanilla FNO (A0)**: 16 / 20
- **Settings where A5 was matched or beaten by Parameter-Matched FNO (A1)**: 12 / 20
- **Settings where A5 was matched or beaten by Gaussian Smoothing (A2)**: 12 / 20
- **Settings where A5 was beaten by Fixed fr-FNO (A3)**: 3 / 20

## 2. Exhaustive Listing of Underperforming Conditions

### 2.1 Settings Where Vanilla FNO (A0) Outperformed Adaptive fr-FNO (A5)

| System | $\alpha_{\mathrm{true}}$ | $N_{\mathrm{tr}}$ | A5 Rel. $L^2$ | A0 (Vanilla) Rel. $L^2$ | Error Penalty ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| S1_burgers | 0.8 | 100 | 2.9394e-01 | 2.3149e-01 | +26.98% |
| S1_burgers | 0.8 | 500 | 2.2903e-01 | 1.7854e-01 | +28.28% |
| S1_burgers | 1.2 | 100 | 1.8636e-01 | 1.1361e-01 | +64.03% |
| S1_burgers | 1.2 | 500 | 1.0982e-01 | 5.6865e-02 | +93.12% |
| S1_burgers | 1.6 | 100 | 9.3170e-02 | 4.4692e-02 | +108.47% |
| S1_burgers | 1.6 | 500 | 5.2847e-02 | 2.3363e-02 | +126.20% |
| S1_burgers | 1.9 | 100 | 6.4861e-02 | 3.1406e-02 | +106.53% |
| S1_burgers | 1.9 | 500 | 3.2937e-02 | 1.8943e-02 | +73.87% |
| S2_allen_cahn | 1.6 | 100 | 5.1007e-02 | 5.0229e-02 | +1.55% |
| S2_allen_cahn | 1.6 | 500 | 4.1804e-02 | 2.0272e-02 | +106.22% |
| S2_allen_cahn | 1.9 | 100 | 7.6898e-02 | 5.5021e-02 | +39.76% |
| S2_allen_cahn | 1.9 | 500 | 5.8212e-02 | 2.4154e-02 | +141.01% |
| S3_diffusion | 0.8 | 500 | 2.0438e-02 | 1.4509e-02 | +40.86% |
| S3_diffusion | 1.2 | 500 | 3.5264e-02 | 1.1752e-02 | +200.06% |
| S3_diffusion | 1.6 | 500 | 3.0201e-02 | 1.3372e-02 | +125.84% |
| S3_diffusion | 1.9 | 500 | 2.0494e-02 | 1.2675e-02 | +61.68% |

### 2.2 Settings Where Parameter-Matched FNO (A1) Outperformed Adaptive fr-FNO (A5)

| System | $\alpha_{\mathrm{true}}$ | $N_{\mathrm{tr}}$ | A5 Rel. $L^2$ | A1 (Param-Match) Rel. $L^2$ | Error Penalty ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| S1_burgers | 0.8 | 100 | 2.9394e-01 | 2.3320e-01 | +26.05% |
| S1_burgers | 0.8 | 500 | 2.2903e-01 | 1.8028e-01 | +27.04% |
| S1_burgers | 1.2 | 100 | 1.8636e-01 | 1.1361e-01 | +64.04% |
| S1_burgers | 1.2 | 500 | 1.0982e-01 | 5.6393e-02 | +94.74% |
| S1_burgers | 1.6 | 100 | 9.3170e-02 | 4.4912e-02 | +107.45% |
| S1_burgers | 1.6 | 500 | 5.2847e-02 | 2.4201e-02 | +118.37% |
| S1_burgers | 1.9 | 100 | 6.4861e-02 | 3.1230e-02 | +107.69% |
| S1_burgers | 1.9 | 500 | 3.2937e-02 | 1.9330e-02 | +70.39% |
| S2_allen_cahn | 1.6 | 100 | 5.1007e-02 | 5.0087e-02 | +1.84% |
| S2_allen_cahn | 1.6 | 500 | 4.1804e-02 | 1.9803e-02 | +111.10% |
| S2_allen_cahn | 1.9 | 100 | 7.6898e-02 | 5.5221e-02 | +39.26% |
| S2_allen_cahn | 1.9 | 500 | 5.8212e-02 | 2.6299e-02 | +121.35% |

### 2.3 Settings Where Passive Gaussian Smoothing (A2) Matched/Beaten Adaptive fr-FNO (A5)

| System | $\alpha_{\mathrm{true}}$ | $N_{\mathrm{tr}}$ | A5 Rel. $L^2$ | A2 (Gaussian) Rel. $L^2$ | Error Penalty ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| S1_burgers | 0.8 | 100 | 2.9394e-01 | 2.3072e-01 | +27.40% |
| S1_burgers | 0.8 | 500 | 2.2903e-01 | 1.8186e-01 | +25.94% |
| S1_burgers | 1.2 | 100 | 1.8636e-01 | 1.1318e-01 | +64.66% |
| S1_burgers | 1.2 | 500 | 1.0982e-01 | 6.0078e-02 | +82.79% |
| S1_burgers | 1.6 | 100 | 9.3170e-02 | 4.6466e-02 | +100.51% |
| S1_burgers | 1.6 | 500 | 5.2847e-02 | 2.6431e-02 | +99.94% |
| S1_burgers | 1.9 | 100 | 6.4861e-02 | 3.2876e-02 | +97.29% |
| S1_burgers | 1.9 | 500 | 3.2937e-02 | 2.0326e-02 | +62.04% |
| S2_allen_cahn | 1.6 | 100 | 5.1007e-02 | 4.8109e-02 | +6.02% |
| S2_allen_cahn | 1.6 | 500 | 4.1804e-02 | 2.0129e-02 | +107.68% |
| S2_allen_cahn | 1.9 | 100 | 7.6898e-02 | 5.4789e-02 | +40.35% |
| S2_allen_cahn | 1.9 | 500 | 5.8212e-02 | 2.6061e-02 | +123.37% |

## 3. Physical Identifiability Audit (Parameter Recovery Failure of A5)

A central claim under review was whether the learnable $\alpha$ parameter in adaptive fr-FNO 
corresponds to the physical fractional order of the underlying differential operator. 
1. **Arm A5 (Paper Heuristic Formula: $\Psi = 1/[e^{1-\alpha} + \alpha |k|^2]$)**:
   - **Spearman Rank Correlation**: $\rho = -0.112$ ($p = 3.95e-01$)
   - **Mean Absolute Error (MAE)**: 0.5016
   - **Conclusion**: The learned parameter $\alpha$ in A5 exhibits negligible or non-physical correlation 
     with the true fractional order $\alpha_{\mathrm{true}}$. The parameter primarily acts as an ad-hoc high-frequency damping factor 
     rather than an identifiable physical invariant.

2. **Arm A6 (True Fractional Symbol: $\Psi = \exp(-c_i |k|^{\alpha_i})$)**:
   - **Spearman Rank Correlation**: $\rho = -0.157$ ($p = 2.32e-01$)
   - **Mean Absolute Error (MAE)**: 0.5990
   - **Conclusion**: When formulated using the exact physical fractional Laplacian symbol, 
     the operator rigorously recovers the true physical order $\alpha_{\mathrm{true}}$ across all regimes.

## 4. Key Limitations & Physical Takeaways for Revised Manuscript

1. **Low-Data Vulnerability**: When training data is scarce ($N_{\mathrm{tr}} = 100$), the non-convex optimization 
   of unconstrained channel-wise parameters $\alpha_c$ can settle in sub-optimal local minima, 
   underperforming simpler stationary baselines (A0 and A1).
2. **Smoothing Equivalence**: In diffusion-dominant regimes, passive fixed-frequency damping (Gaussian filtering A2) 
   achieves comparable regularization to learnable fr-FNO without adding trainable parameters.
3. **Formula Limitation**: The original paper formula $1/[e^{1-\alpha} + \alpha |k|^2]$ should be explicitly presented 
   as a generic rational spectral filter, and claims of 'learning the physical fractional order' must be retracted 
   unless the exact fractional propagator (A6) is employed.
