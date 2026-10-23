import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'figure.autolayout': True,
})

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), dpi=300)

k = np.geomspace(1, 20, 200)
alphas = [0.8, 1.2, 1.6, 2.0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Left: Rational Filter Psi(k) = 1 / (exp(1-alpha) + alpha * k^2)
ax1 = axes[0]
for alpha, c in zip(alphas, colors):
    psi_rat = 1.0 / (np.exp(1.0 - alpha) + alpha * k**2)
    ax1.plot(k, psi_rat, label=f'$\\alpha = {alpha}$', color=c, lw=1.8)

# Add reference guide line for k^-2
k_ref = np.geomspace(5, 18, 50)
ref_line = 3.0 * k_ref**(-2.0)
ax1.plot(k_ref, ref_line, 'k--', lw=1.2, alpha=0.7, label='Slope $-2$ (ref)')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Wavenumber $k$ (log scale)')
ax1.set_ylabel('Spectral Filter $\\Psi(k)$ (log scale)')
ax1.set_title('Rational Filter: $\\Psi(k) = [e^{1-\\alpha} + \\alpha k^2]^{-1}$\n(Asymptotic slope is $-2$ for all $\\alpha$)', fontsize=8.5)
ax1.grid(True, which='both', ls=':', alpha=0.5)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')

# Right: Fractional Symbol: -log Psi(k) = beta * k^alpha
ax2 = axes[1]
beta = 0.1
for alpha, c in zip(alphas, colors):
    neg_log_psi = beta * k**alpha
    ax2.plot(k, neg_log_psi, label=f'$\\alpha = {alpha}$ (slope ${alpha}$)', color=c, lw=1.8)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Wavenumber $k$ (log scale)')
ax2.set_ylabel('$-\\log\\,\\Psi(k)$ (log scale)')
ax2.set_title('Fractional Symbol: $\\Psi(k) = \\exp(-\\beta |k|^\\alpha)$\n(Log-log slope of $-\\log\\,\\Psi$ is $\\alpha$)', fontsize=8.5)
ax2.grid(True, which='both', ls=':', alpha=0.5)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

plt.savefig('figures/fig_spectral_decay_comparison.pdf', bbox_inches='tight')
print("Successfully generated figures/fig_spectral_decay_comparison.pdf")
