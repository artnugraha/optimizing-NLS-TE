# main.py
import numpy as np
import scipy.constants as sc

from integrator import TwoBandTEModelNotebook
from plot import plot_velocity_panel_2x3


def main():
    # Notebook parameters
    T = 300.0
    tau = 1e-14
    infty = 100.0

    model = TwoBandTEModelNotebook(T=T, tau=tau, infty=infty)

    npoint = 301
    eta = np.linspace(-15, 15, npoint)

    # Notebook: e0 = 0.26/kBT
    e0 = 0.26 / model.kBT

    # Velocities (notebook cell)
    vF1 = 1.89e5
    vF2 = 5.17e5
    vF3 = 1.03e5

    # Effective mass (match notebook usage)
    c_light = sc.physical_constants["speed of light in vacuum"][0]
    m0 = sc.physical_constants["electron mass energy equivalent in MeV"][0] / (c_light**2)
    eff_mass1 = 0.4 * m0 * 1e6

    # Energy shifts (notebook cell)
    eshift1 = 6.5
    eshift2 = 9.0
    eshift3 = 4.5

    plot_velocity_panel_2x3(
        model=model,
        eta=eta,
        e0=e0,
        vF_list=(vF1, vF2, vF3),
        eff_mass=eff_mass1,
        eshift_list=(eshift1, eshift2, eshift3),
        band_data=None,  # keep None unless you wire band-structure arrays
        outpath="2BM_typeI_velocity.png",
    )


if __name__ == "__main__":
    main()

