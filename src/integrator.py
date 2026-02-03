# integrator.py
import numpy as np
from scipy.integrate import quad
import scipy.constants as sc


class TwoBandTEModelNotebook:
    """
    Based on the integrator + other functions
    """

    def __init__(self, T, tau, infty=100.0):
        self.T = float(T)
        self.tau = float(tau)
        self.infty = float(infty)

        # Notebook constants 
        self.kB = sc.physical_constants["Boltzmann constant in eV/K"][0]
        self.hbar = sc.physical_constants["reduced Planck constant in eV s"][0]
        self.e_charge = sc.physical_constants["atomic unit of charge"][0]
        self.pi = np.pi

        self.kBT = self.kB * self.T
        self.S_0 = self.kB  # notebook: S_0 = kB*e_charge/e_charge

    # -------------------------
    # Notebook prefactors
    # -------------------------
    def Sigma0c(self, vF):
        frac1c = self.tau * self.e_charge**2 * self.kBT**2
        return frac1c / (3 * self.pi**2 * self.hbar**3 * vF * self.e_charge)

    def Sigma0v(self, eff_mass):
        frac1v = 2 * self.tau * self.e_charge**2 * self.kBT / (3 * self.pi**2 * self.hbar**2)
        frac2v = 2 * eff_mass * self.kBT / (self.hbar**2)
        rootv = np.sqrt(frac2v)
        return frac1v * rootv / self.e_charge

    def Kappa0c(self, vF):
        return self.tau * self.kB**4 * self.T**3 / (3 * self.pi**2 * self.hbar**3 * vF) * self.e_charge

    def Kappa0v(self, eff_mass):
        fracv = 2 * eff_mass * self.kBT / (self.hbar**2)
        rootv = np.sqrt(fracv)
        return 2 * self.tau * self.kB**3 * self.T**2 / (3 * self.pi**2 * self.hbar**2) * rootv * self.e_charge

    # -------------------------
    # Notebook FD derivative (raw exp form)
    # -------------------------
    @staticmethod
    def fd_deriv_exp(x):
        ex = np.exp(x)
        return ex / (ex + 1.0) ** 2

    # -------------------------
    # Integral kernel
    # -------------------------
    def L_integral(self, i, a, band="c", E0=None):
        """
        Matches notebook bounds:
          conduction: x in [-a, +infty]
          valence:    x in [-infty, E0-a]
        """
        if band == "c":
            def integrand(x):
                return (x + a) ** 2 * (x ** i) * self.fd_deriv_exp(x)

            lo, hi = -a, self.infty

        elif band == "v":
            if E0 is None:
                raise ValueError("E0 required for valence band")

            def integrand(x):
                return (-x - a + E0) ** 1.5 * (x ** i) * self.fd_deriv_exp(x)

            lo, hi = -self.infty, (E0 - a)

        else:
            raise ValueError("band must be 'c' or 'v'")

        # quad defaults (notebook did not override epsrel/epsabs/limit)
        val, _ = quad(integrand, lo, hi)
        return val

    # -------------------------
    # Conduction transport
    # -------------------------
    def sigma_c(self, x, E0, vF):
        return np.array([self.L_integral(0, xi, "c", E0) for xi in x]) * self.Sigma0c(vF)

    def S_c(self, x, E0):
        return np.array([
            self.L_integral(1, xi, "c", E0) / self.L_integral(0, xi, "c", E0)
            for xi in x
        ]) * self.S_0

    def kappa_c(self, x, E0, vF):
        return np.array([
            self.L_integral(2, xi, "c", E0)
            - (self.L_integral(1, xi, "c", E0) ** 2) / self.L_integral(0, xi, "c", E0)
            for xi in x
        ]) * self.Kappa0c(vF)

    # -------------------------
    # Valence transport
    # -------------------------
    def sigma_v(self, x, E0, eff_mass):
        return np.array([self.L_integral(0, xi, "v", E0) for xi in x]) * self.Sigma0v(eff_mass)

    def S_v(self, x, E0):
        return np.array([
            self.L_integral(1, xi, "v", E0) / self.L_integral(0, xi, "v", E0)
            for xi in x
        ]) * self.S_0

    def kappa_v(self, x, E0, eff_mass):
        return np.array([
            self.L_integral(2, xi, "v", E0)
            - (self.L_integral(1, xi, "v", E0) ** 2) / self.L_integral(0, xi, "v", E0)
            for xi in x
        ]) * self.Kappa0v(eff_mass)

    # -------------------------
    # Constants
    # -------------------------
    def sigma(self, x, E0, vF, eff_mass):
        return self.sigma_c(x, E0, vF) + self.sigma_v(x, E0, eff_mass)

    def S(self, x, E0, vF, eff_mass):
        sc = self.sigma_c(x, E0, vF)
        sv = self.sigma_v(x, E0, eff_mass)
        return (sc * self.S_c(x, E0) + sv * self.S_v(x, E0)) / (sc + sv)

    def kappa(self, x, E0, vF, eff_mass):
        sc = self.sigma_c(x, E0, vF)
        sv = self.sigma_v(x, E0, eff_mass)
        return (
            self.kappa_c(x, E0, vF)
            + self.kappa_v(x, E0, eff_mass)
            + (sc * sv * (self.S_c(x, E0) - self.S_v(x, E0)) ** 2) / (sc + sv) * self.T
        )

    def PF(self, x, E0, vF, eff_mass):
        return self.S(x, E0, vF, eff_mass) ** 2 * self.sigma(x, E0, vF, eff_mass)

    def ZT(self, x, E0, vF, eff_mass):
        return self.PF(x, E0, vF, eff_mass) / self.kappa(x, E0, vF, eff_mass)

