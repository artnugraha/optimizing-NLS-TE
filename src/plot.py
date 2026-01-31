# plot.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def apply_notebook_style():
    mpl.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 22
    plt.rcParams["axes.linewidth"] = 2


def plot_velocity_panel_2x3(
    model,
    eta,
    e0,
    vF_list,
    eff_mass,
    eshift_list,
    band_data=None,
    outpath="2BM_typeI_velocity.png",
):
    """
    Matches notebook panel layout and normalization:
      ax2: S/S0
      ax3: sigma/sigma0
      ax4: kappa/kappa0
      ax5: PF/(S0^2*sigma0)
      ax6: ZT*T
    """

    apply_notebook_style()
    fig = plt.figure(figsize=(12, 8))

    # Unpack
    vF1, vF2, vF3 = vF_list
    es1, es2, es3 = eshift_list

    # Normalizations (notebook)
    sigma0 = model.Sigma0v(eff_mass)
    kappa0 = model.Kappa0v(eff_mass)
    S0 = model.S_0
    T = model.T

    # (a) Optional dispersion plot
    ax1 = plt.subplot(2, 3, 1)
    if band_data is not None:
        kcc, fc, fc1, fc2, fc3, Ekcc = band_data
        ax1.plot(kcc, fc, linestyle="solid", color="red", label=r"$D_{1}$")
        ax1.plot(kcc, fc1, linestyle="dashed", color="green", label=r"$D_{2}$")
        ax1.plot(kcc, fc3, linestyle="dashdot", color="blue", label=r"$D_{3}$")
        ax1.set_xlabel(r"$k (2\pi/a)$")
        ax1.set_ylabel(r"$E~(\mathrm{eV})$")
        ax1.axvline(0, color="grey", linestyle="-", linewidth=1)
    else:
        ax1.axis("off")

    # (b) Seebeck
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(eta, model.S(eta + es1, e0, vF1, eff_mass) / S0, color="red", linestyle="solid")
    ax2.plot(eta, model.S(eta + es2, e0, vF2, eff_mass) / S0, color="green", linestyle="dashed")
    ax2.plot(eta, model.S(eta + es3, e0, vF3, eff_mass) / S0, color="blue", linestyle="dashdot")
    ax2.axis([-15, 15, -1, 1])
    ax2.axvline(0, color="grey", linestyle="-", linewidth=1)
    ax2.set_xlabel(r"$\mu/k_BT$")
    ax2.set_ylabel(r"$S$ $(S_0)$")
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.labelpad = -4
    ax2.xaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax2.xaxis.set_tick_params(which="minor", size=7, width=2, direction="in")
    ax2.yaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax2.yaxis.set_tick_params(which="minor", size=7, width=2, direction="in")

    # (c) Electrical conductivity
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(eta, model.sigma(eta + es1, e0, vF1, eff_mass) / sigma0, color="red", linestyle="solid")
    ax3.plot(eta, model.sigma(eta + es2, e0, vF2, eff_mass) / sigma0, color="green", linestyle="dashed")
    ax3.plot(eta, model.sigma(eta + es3, e0, vF3, eff_mass) / sigma0, color="blue", linestyle="dashdot")
    ax3.axis([-15, 15, 0, 30])
    ax3.axvline(0, color="grey", linestyle="-", linewidth=1)
    ax3.set_xlabel(r"$\mu/k_BT$")
    ax3.set_ylabel(r"$\sigma (\sigma_0)$")
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.labelpad = -4
    ax3.xaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax3.xaxis.set_tick_params(which="minor", size=7, width=2, direction="in")
    ax3.yaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax3.yaxis.set_tick_params(which="minor", size=7, width=2, direction="in")

    # (d) Thermal conductivity
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(eta, model.kappa(eta + es1, e0, vF1, eff_mass) / kappa0, color="red", linestyle="solid")
    ax4.plot(eta, model.kappa(eta + es2, e0, vF2, eff_mass) / kappa0, color="green", linestyle="dashed")
    ax4.plot(eta, model.kappa(eta + es3, e0, vF3, eff_mass) / kappa0, color="blue", linestyle="dashdot")
    ax4.axis([-15, 15, 0, 200])
    ax4.axvline(0, color="grey", linestyle="-", linewidth=1)
    ax4.set_xlabel(r"$\mu/k_BT$")
    ax4.set_ylabel(r"$\kappa$ $(\kappa_0)$")
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.labelpad = -4
    ax4.xaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax4.xaxis.set_tick_params(which="minor", size=7, width=2, direction="in")
    ax4.yaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax4.yaxis.set_tick_params(which="minor", size=7, width=2, direction="in")

    # (e) Power Factor
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(eta, model.PF(eta + es1, e0, vF1, eff_mass) / (S0**2 * sigma0), color="red", linestyle="solid")
    ax5.plot(eta, model.PF(eta + es2, e0, vF2, eff_mass) / (S0**2 * sigma0), color="green", linestyle="dashed")
    ax5.plot(eta, model.PF(eta + es3, e0, vF3, eff_mass) / (S0**2 * sigma0), color="blue", linestyle="dashdot")
    ax5.axis([-15, 15, 0, 20])
    ax5.axvline(0, color="grey", linestyle="-", linewidth=1)
    ax5.set_xlabel(r"$\mu/k_BT$")
    ax5.set_ylabel(r"$\mathrm{PF}~(S_0^2 \sigma_0)$")
    ax5.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax5.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax5.yaxis.labelpad = -4
    ax5.xaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax5.xaxis.set_tick_params(which="minor", size=7, width=2, direction="in")
    ax5.yaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax5.yaxis.set_tick_params(which="minor", size=7, width=2, direction="in")

    # (f) ZT_e (notebook plots ZT*T)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(eta, model.ZT(eta + es1, e0, vF1, eff_mass) * T, color="red", linestyle="solid")
    ax6.plot(eta, model.ZT(eta + es2, e0, vF2, eff_mass) * T, color="green", linestyle="dashed")
    ax6.plot(eta, model.ZT(eta + es3, e0, vF3, eff_mass) * T, color="blue", linestyle="dashdot")
    ax6.axis([-15, 15, 0, 0.15])
    ax6.axvline(0, color="grey", linestyle="-", linewidth=1)
    ax6.set_xlabel(r"$\mu/k_BT$")
    ax6.set_ylabel(r"ZT$_{e}$")
    ax6.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax6.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax6.yaxis.labelpad = -4
    ax6.xaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax6.xaxis.set_tick_params(which="minor", size=7, width=2, direction="in")
    ax6.yaxis.set_tick_params(which="major", size=10, width=2, direction="in")
    ax6.yaxis.set_tick_params(which="minor", size=7, width=2, direction="in")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

