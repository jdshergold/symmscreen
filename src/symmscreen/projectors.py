"""Crystal and molecule point-group projectors, Pi_l^(L) and Pi_l^(H).

See Sec. "Crystal space groups and anisotropy" and Sec. "Molecular crystals" of the
paper for the definitions:

    Pi_l = (1/N) sum_i p_{i,l} G^(l)(tilde R_i),   p_{i,l} = det(R_i)^l,   tilde R_i = det(R_i) R_i

Point-group operations obtained from a Hermann-Mauguin/Schoenflies symbol (via
`pg_ops_from_sg_number`, `pg_ops_from_sg_symbol`, or `pg_ops_from_pg_symbol`) come out
in pymatgen's conventional crystallographic axes, which for some point groups (e.g.
trigonal/hexagonal) are not orthonormal. These must be transformed into genuine
Cartesian O(3) matrices via a lattice matrix A before evaluating the Wigner-G matrices:
`R_cart = A @ R @ inv(A)`. Operations obtained directly from atomic coordinates (via
`pg_ops_from_atoms`) are already expressed in genuine Cartesian coordinates and need no
such transform. This module keeps that distinction explicit rather than hiding it.
"""

import numpy as np

from ._clean import clean_matrix
from ._wigner import wigner_g
from .cif import cif_crystal_info
from .lattice import crystal_lattice_matrix, pure_lattice_matrix
from .point_groups import MOLECULE_SYMMETRY_TOLERANCE, pg_ops


def _project_ops(l, ops, lattice_matrix=None):
    """Pi_l = (1/N) sum_i p_{i,l} G^(l)(tilde R_i), optionally changing basis first."""
    if lattice_matrix is not None:
        inv_A = np.linalg.inv(lattice_matrix)
        ops = [lattice_matrix @ op @ inv_A for op in ops]
    ps = np.array([np.linalg.det(op) ** l for op in ops])
    Gs = np.array([wigner_g(l, np.linalg.det(op) * op) for op in ops])
    return clean_matrix(sum(p * G for p, G in zip(ps, Gs)) / len(ops))


def _xi_inf_from_ops(ops):
    """Analytic asymptotic even-l survival fraction: (1 + alpha)/N.

    alpha = 1 if the group contains inversion (which, like the identity, contributes
    fully at every l), 0 otherwise.
    """
    n = len(ops)
    alpha = 1 if any(np.allclose(op, -np.eye(3), atol=1e-5) for op in ops) else 0
    return (1 + alpha) / n


class CrystalProjector:
    """Crystallographic point-group projector Pi_l^(L).

    Construct from exactly one of `sg_name`, `sg_number`, or `pg_symbol`. Pass real
    lattice vectors `a`, `b`, `c` (Angstrom, as 3-vectors) to work in the actual
    crystal's Cartesian frame; otherwise an abstract Cartesian frame is built purely
    from the point-group metric (basis-independent quantities such as `kappa` and
    `xi` do not depend on this choice, see Appendix "Invariance of anisotropy
    measures" in the paper).
    """

    def __init__(self, sg_name=None, sg_number=None, pg_symbol=None, a=None, b=None, c=None):
        n_given = sum(x is not None for x in (sg_name, sg_number, pg_symbol))
        if n_given != 1:
            raise ValueError("Provide exactly one of sg_name, sg_number, or pg_symbol.")
        self.sg_name = sg_name
        self.sg_number = sg_number
        self.pg_symbol = pg_symbol
        self.a, self.b, self.c = a, b, c
        self._has_lattice = all(v is not None for v in (a, b, c))
        self._ops = None
        self._cache = {}

    @property
    def ops(self):
        if self._ops is None:
            self._ops = pg_ops(sg_name=self.sg_name, sg_number=self.sg_number, pg_symbol=self.pg_symbol)
        return self._ops

    def _lattice_matrix(self):
        if self._has_lattice:
            return crystal_lattice_matrix(self.a, self.b, self.c)
        return pure_lattice_matrix(sg_name=self.sg_name, sg_number=self.sg_number, pg_symbol=self.pg_symbol)

    def matrix(self, l):
        """Pi_l^(L), cached per `l`."""
        if l not in self._cache:
            self._cache[l] = _project_ops(l, self.ops, lattice_matrix=self._lattice_matrix())
        return self._cache[l]

    def kappa(self, l):
        """Fractional anisotropy survival kappa_l^(L) = tr(Pi_l)/(2l+1)."""
        return float(np.trace(self.matrix(l)) / (2 * l + 1))

    def xi(self, l_max):
        """Cumulative even-l survival fraction up to `l_max` (must be even)."""
        norm = l_max * (l_max + 3) // 2
        if norm <= 0:
            return 0.0
        total = sum(np.trace(self.matrix(l)) for l in range(2, l_max + 1, 2))
        return float(total / norm)

    @property
    def xi_inf(self):
        """Analytic asymptotic even-l survival fraction (l -> infinity)."""
        return _xi_inf_from_ops(self.ops)

    @property
    def quadrupole_class(self):
        """Quadrupole survival class k = rank(Pi_2) in {0, 1, 2, 3, 5} (paper's Q_k)."""
        return int(round(5 * self.kappa(2)))

    @classmethod
    def from_cif(cls, cif_path):
        """Build from a CIF's space group and lattice vectors."""
        sg_name, a, b, c, _atoms = cif_crystal_info(cif_path)
        return cls(sg_name=sg_name, a=a, b=b, c=c)


class MoleculeProjector:
    """Internal molecular point-group projector Pi_l^(H).

    Construct from exactly one of `atoms` (list of `[symbol, x, y, z]`) or `pg_symbol`
    (an idealised molecule with no coordinates, only a symmetry label).
    """

    def __init__(self, atoms=None, pg_symbol=None, tolerance=None):
        if (atoms is None) == (pg_symbol is None):
            raise ValueError("Provide exactly one of atoms or pg_symbol.")
        self.atoms = atoms
        self.pg_symbol = pg_symbol
        self.tolerance = MOLECULE_SYMMETRY_TOLERANCE if tolerance is None else tolerance
        self._ops = None
        self._cache = {}

    @property
    def ops(self):
        if self._ops is None:
            if self.atoms is not None:
                self._ops = pg_ops(atoms=self.atoms, tolerance=self.tolerance)
            else:
                self._ops = pg_ops(pg_symbol=self.pg_symbol)
        return self._ops

    def _internal_matrix(self, l):
        """Pi_l^(H) in the molecule's own frame (unrotated)."""
        if l not in self._cache:
            if self.atoms is not None:
                # Atomic coordinates are already genuine Cartesian: no basis change.
                lattice_matrix = None
            else:
                # Symbol-only ops come out in pymatgen's (possibly non-orthogonal)
                # conventional axes and need the same basis change as a crystal's.
                lattice_matrix = pure_lattice_matrix(pg_symbol=self.pg_symbol)
            self._cache[l] = _project_ops(l, self.ops, lattice_matrix=lattice_matrix)
        return self._cache[l]

    def matrix(self, l, T=None):
        """Pi_l^(H), or Pi_{l,T}^(H) = G(T) Pi_l^(H) G(T)^T if a rotation `T` is given.

        `T` (\\mathcal{T} in the paper) maps the molecule's internal frame onto its
        embedding in some external (e.g. crystal) frame.
        """
        Pi = self._internal_matrix(l)
        if T is None:
            return Pi
        G_T = wigner_g(l, T)
        return clean_matrix(G_T @ Pi @ G_T.T)

    def kappa(self, l, T=None):
        """Fractional anisotropy survival kappa_l^(H) = tr(Pi_l)/(2l+1)."""
        return float(np.trace(self.matrix(l, T=T)) / (2 * l + 1))

    def xi(self, l_max, T=None):
        """Cumulative even-l survival fraction up to `l_max` (must be even)."""
        norm = l_max * (l_max + 3) // 2
        if norm <= 0:
            return 0.0
        total = sum(np.trace(self.matrix(l, T=T)) for l in range(2, l_max + 1, 2))
        return float(total / norm)

    @property
    def xi_inf(self):
        """Analytic asymptotic even-l survival fraction (l -> infinity)."""
        return _xi_inf_from_ops(self.ops)

    @property
    def quadrupole_class(self):
        """Quadrupole survival class k = rank(Pi_2) in {0, 1, 2, 3, 5} (paper's Q_k),
        for the molecule's own internal symmetry H (unrotated, i.e. T = identity)."""
        return int(round(5 * self.kappa(2)))

    @classmethod
    def from_cif(cls, cif_path, molecule_index=0):
        """Build from one molecule's atoms in a CIF (already in the crystal frame)."""
        _sg, _a, _b, _c, atoms = cif_crystal_info(cif_path, molecule_index=molecule_index)
        return cls(atoms=atoms)
