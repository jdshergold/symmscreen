"""Real Wigner-G rotation matrices, in the real-spherical-harmonic basis.

This is a trimmed extraction of the ``WignerG`` matrix construction from
Ben Lillard's `vsdm` package (https://github.com/blillard/vsdm, MIT License),
which builds the real analogue of the Wigner-D matrix from the complex
Wigner-D matrix supplied by the `spherical` package. We vendor just this
construction so `symmscreen` does not need `vsdm`'s full set pf dependencies
(`vegas`, `h5py`), which are unused for symmetry-projector work.

Convention: for an active rotation R with z-y-z Euler angles
(alpha, beta, gamma), G^(l)_{m'm}(R) is defined from the Wigner "little d"
matrix via D^(l)_{m'm} = exp(-i m' alpha) d^(l)_{m'm}(beta) exp(-i m gamma),
matching the convention used by SCarFFF/VSDM form-factor coefficients.
"""

from functools import lru_cache

import numpy as np
import quaternionic
import spherical
from scipy.spatial.transform import Rotation

from ._clean import clean_matrix

_WIGNER_G_CACHE = {}


class WignerG:
    """Assembles the real form of the Wigner-D matrix for l = 0..l_max."""

    def __init__(self, l_max):
        self.wigD = spherical.Wigner(l_max)
        self.l_max = l_max
        # spherical >= 1.0 returns the complex conjugate of the D(R) convention
        # documented above.
        self.conj_D = True

    def G_l(self, R):
        """Return {l: (2*l+1, 2*l+1) real matrix} for l = 0..l_max."""
        mxD = np.conjugate(self.wigD.D(R)) if self.conj_D else self.wigD.D(R)
        gL = {}
        for l in range(self.l_max + 1):
            if l == 0:
                d_00 = mxD[self.wigD.Dindex(0, 0, 0)]
                gL[0] = np.array([[np.real(d_00)]])
                continue

            d_00 = mxD[self.wigD.Dindex(l, 0, 0)]
            d_0p_k = np.array(
                [[(-1) ** k * mxD[self.wigD.Dindex(l, 0, k)] for k in range(1, l + 1)]]
            )
            d_p0_j = np.array(
                [[(-1) ** j * mxD[self.wigD.Dindex(l, j, 0)]] for j in range(1, l + 1)]
            )
            d_mp_jk = np.array(
                [
                    [(-1) ** k * mxD[self.wigD.Dindex(l, -j, k)] for k in range(1, l + 1)]
                    for j in range(1, l + 1)
                ]
            )
            d_pp_jk = np.array(
                [
                    [(-1) ** (j + k) * mxD[self.wigD.Dindex(l, j, k)] for k in range(1, l + 1)]
                    for j in range(1, l + 1)
                ]
            )

            G_mm_jk = np.real(d_pp_jk) - np.real(d_mp_jk)
            G_mp_jk = -np.imag(d_pp_jk) + np.imag(d_mp_jk)
            G_pm_jk = np.imag(d_pp_jk) + np.imag(d_mp_jk)
            G_pp_jk = np.real(d_pp_jk) + np.real(d_mp_jk)
            G_m0_j = -np.sqrt(2) * np.imag(d_p0_j)
            G_p0_j = np.sqrt(2) * np.real(d_p0_j)
            G_0m_k = np.sqrt(2) * np.imag(d_0p_k)
            G_0p_k = np.sqrt(2) * np.real(d_0p_k)
            G_00 = np.real(d_00)

            G_mm = np.flip(np.flip(G_mm_jk, axis=1), axis=0)
            G_mp = np.flip(G_mp_jk, axis=0)
            G_pm = np.flip(G_pm_jk, axis=1)
            G_m0 = np.flip(G_m0_j, axis=0)
            G_0m = np.flip(G_0m_k, axis=1)

            _G_m = np.concatenate((G_mm, G_m0, G_mp), axis=1)
            _G_0 = np.concatenate((G_0m, [[G_00]], G_0p_k), axis=1)
            _G_p = np.concatenate((G_pm, G_p0_j, G_pp_jk), axis=1)
            gL[l] = np.concatenate((_G_m, _G_0, _G_p), axis=0)
        return gL


def _as_quaternionic(R):
    """Convert a proper 3x3 rotation matrix to quaternionic's (w, x, y, z) order."""
    quat_xyzw = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return quaternionic.array(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])


@lru_cache(maxsize=20000)
def _wigner_g_cached(l, R_key):
    R = np.asarray(R_key, dtype=float).reshape(3, 3)
    quat = _as_quaternionic(R)
    G = _WIGNER_G_CACHE[l].G_l(quat)[l]
    return clean_matrix(np.asarray(G, dtype=float))


def wigner_g(l, R):
    """Get the real Wigner-G matrix for angular momentum `l` and proper rotation `R`."""
    if l not in _WIGNER_G_CACHE:
        _WIGNER_G_CACHE[l] = WignerG(l)

    R_key = tuple(np.round(np.asarray(R, dtype=float).reshape(9), decimals=12))
    return _wigner_g_cached(l, R_key)
