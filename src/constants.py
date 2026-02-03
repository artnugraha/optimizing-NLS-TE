# constants.py
import numpy as np

e = 1.602176634e-19
kB = 1.380649e-23
hbar = 1.054571817e-34

S_0 = kB / e


def Sigma0c(vF):
    # EXACT expression from notebook
    return e**2 / (3 * np.pi**2 * hbar) * vF


def Sigma0v(m_eff):
    # EXACT expression from notebook
    return e**2 / (3 * np.pi**2 * hbar) * np.sqrt(m_eff)


def Kappa0c(vF):
    return kB**2 / (3 * np.pi**2 * hbar) * vF


def Kappa0v(m_eff):
    return kB**2 / (3 * np.pi**2 * hbar) * np.sqrt(m_eff)

