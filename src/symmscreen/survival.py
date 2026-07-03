"""Combined molecule-in-crystal survival operator, and the Lambda modulation-loss estimators.

See Sec. "Molecular crystals" and Sec. "Modulation signals in dark matter detectors" of
the paper. The combined survival operator is

    C^(l)(T) = Pi_l^(L) @ G^(l)(T) Pi_l^(H) G^(l)(T)^T,

with `T` the rotation mapping the molecule's internal frame onto its embedding in the
crystal (called \\mathcal{T} in the paper). Unlike Pi_l^(L) or Pi_l^(H) alone, C^(l) is
not itself a projector, so its trace (rather than its rank) measures the fraction of
anisotropy that survives.
"""

import numpy as np

from ._clean import clean_matrix
from .cif import cif_crystal_info
from .projectors import CrystalProjector, MoleculeProjector
from .shape import coordinate_shape_vector


class CombinedSurvival:
    """Combined molecule-in-crystal survival operator C^(l)(T), and its Lambda estimators.

    Parameters
    ----------
    crystal : CrystalProjector
    molecule : MoleculeProjector, optional
        If omitted, C^(l) reduces to the crystal projector Pi_l^(L) (an unconstrained,
        fully anisotropic "molecule").
    T : (3, 3) array, optional
        Rotation from the molecule's internal frame into the crystal frame (\\mathcal{T}
        in the paper). Defaults to the identity, which is already the correct choice
        when `molecule` was built directly from atoms expressed in the crystal frame
        (e.g. via `MoleculeProjector.from_cif` or `CombinedSurvival.from_cif`).
    """

    def __init__(self, crystal, molecule=None, T=None):
        self.crystal = crystal
        self.molecule = molecule
        self.T = np.eye(3) if T is None else np.asarray(T, dtype=float)
        self._cache = {}
        self._l_0_cache = None

    def _Pi_H(self, l):
        if self.molecule is None:
            return np.eye(2 * l + 1)
        return self.molecule.matrix(l, T=self.T)

    def matrix(self, l):
        """Combined survival operator C^(l)(T). Returns Pi_l^(L) if no molecule is set."""
        if l not in self._cache:
            Pi_P = self.crystal.matrix(l)
            if self.molecule is None:
                self._cache[l] = Pi_P
            else:
                self._cache[l] = clean_matrix(Pi_P @ self._Pi_H(l))
        return self._cache[l]

    def kappa(self, l):
        """Fractional anisotropy survival kappa_l(T) = tr(C^(l))/(2l+1)."""
        return float(np.trace(self.matrix(l)) / (2 * l + 1))

    def xi(self, l_max):
        """Cumulative even-l combined survival fraction up to `l_max` (must be even)."""
        norm = l_max * (l_max + 3) // 2
        if norm <= 0:
            return 0.0
        total = sum(np.trace(self.matrix(l)) for l in range(2, l_max + 1, 2))
        return float(total / norm)

    # ==== l_0 and the exponential mode weight w_l^(H). ==== #

    def _l_0(self, l_max):
        """Lowest even l >= 2 with tr(Pi_H^(l)) > 0."""
        if self._l_0_cache is None:
            if self.molecule is None:
                self._l_0_cache = 2
            else:
                for l in range(2, l_max + 1, 2):
                    if np.trace(self.molecule.matrix(l)) > 1e-5:
                        self._l_0_cache = l
                        break
                else:
                    self._l_0_cache = 2
        return self._l_0_cache

    @staticmethod
    def _w(l, l_0):
        if l <= 0:
            return 0.0
        return np.exp(-(l_0 + 1) * np.sqrt(l * (l + 1)) / l_0) / np.sqrt(l)

    # ==== Lambda estimators (no electronic-structure calculation required). ==== #

    def lambda_L(self, l_max=12):
        """Lambda^(L)(T): orientation-specific group-theoretic modulation-loss estimator.

        Ratio of the expected f_RMS for this crystal/molecule/orientation to that of the
        same molecule with trivial (C1) crystal symmetry, estimated purely from the
        surviving angular modes weighted by an exponentially decaying mode importance.
        Requires a molecule (its internal symmetry sets the normalisation).
        """
        if self.molecule is None:
            raise ValueError("lambda_L requires a molecule (internal symmetry H).")
        l_0 = self._l_0(l_max)
        num = den = 0.0
        for l in range(2, l_max + 1, 2):
            w2 = self._w(l, l_0) ** 2 / (2 * l + 1)
            num += w2 * np.trace(self.matrix(l))
            den += w2 * np.trace(self.molecule.matrix(l))
        if den < 1e-12:
            return 0.0
        return float(np.sqrt(max(num / den, 0.0)))

    def lambda_L_avg(self, l_max=12):
        """<Lambda^(L)>_T: analytic SO(3)-average over an unknown molecular orientation.

        Useful when the crystal structure (and hence the molecular embedding `T`) has
        not yet been determined, e.g. when screening candidate molecules before
        synthesis. Uses <tr C^(l)(T)>_T = tr(Pi_P) tr(Pi_H)/(2l+1) (Haar-measure result).
        """
        if self.molecule is None:
            raise ValueError("lambda_L_avg requires a molecule (internal symmetry H).")
        l_0 = self._l_0(l_max)
        num = den = 0.0
        for l in range(2, l_max + 1, 2):
            w2 = self._w(l, l_0) ** 2 / (2 * l + 1)
            kappa_P = self.crystal.kappa(l)
            kappa_H = self.molecule.kappa(l)
            num += w2 * kappa_P * kappa_H
            den += w2 * kappa_H
        if den < 1e-12:
            return 0.0
        return float(np.sqrt(max(num / den, 0.0)))

    def lambda_ideal_avg(self, l_max=12):
        """<Lambda>_T: analytic SO(3)-average of the *absolute* modulation-loss estimator.

        Unlike `lambda_L_avg`, which measures loss relative to the same molecule with
        no crystal (and so ignores whether the molecule itself is symmetric), this
        compares to a hypothetical molecule with no internal symmetry at all (H = C1).
        It is what the paper calls Lambda, an absolute measure of crystal+molecule
        quality useful for ranking *different* candidate molecules against each other
        (a highly symmetric molecule can never score well here, even in a C1 crystal).
        """
        if self.molecule is None:
            raise ValueError("lambda_ideal_avg requires a molecule (internal symmetry H).")
        l_0 = self._l_0(l_max)
        num = den = 0.0
        for l in range(2, l_max + 1, 2):
            w2 = self._w(l, l_0) ** 2 / (2 * l + 1)
            kappa_P = self.crystal.kappa(l)
            kappa_H = self.molecule.kappa(l)
            num += w2 * kappa_P * kappa_H
            den += w2
        if den < 1e-12:
            return 0.0
        return float(np.sqrt(max(num / den, 0.0)))

    def lambda_coord(self, shape_vector=None):
        """Lambda_coord: coordinate-aware l=2 modulation-loss estimator.

        The best-performing estimator in the paper (shown to track the true f_RMS ratio
        linearly, Sec. "Demonstrations"). Requires the molecule's atomic coordinates,
        either via `shape_vector` directly or because `molecule` was built from atoms.
        """
        if shape_vector is None:
            if self.molecule is None or self.molecule.atoms is None:
                raise ValueError(
                    "lambda_coord requires atoms: pass shape_vector explicitly, "
                    "or build the molecule from atoms."
                )
            shape_vector = coordinate_shape_vector(self.molecule.atoms)
        shape_vector = np.asarray(shape_vector, dtype=float)

        numerator = np.linalg.norm(self.matrix(2) @ shape_vector)
        if self.molecule is not None:
            denominator = np.linalg.norm(self._Pi_H(2) @ shape_vector)
        else:
            denominator = np.linalg.norm(shape_vector)
        if denominator < 1e-12:
            return 0.0
        return float(numerator / denominator)

    # ==== Convenience constructors. ==== #

    @classmethod
    def from_cif(cls, cif_path, molecule_index=0):
        """Build crystal + molecule + T directly from one CIF file.

        The molecule's atoms are already expressed in the crystal's Cartesian frame,
        so `T` is the identity.
        """
        sg_name, a, b, c, atoms = cif_crystal_info(cif_path, molecule_index=molecule_index)
        crystal = CrystalProjector(sg_name=sg_name, a=a, b=b, c=c)
        molecule = MoleculeProjector(atoms=atoms)
        return cls(crystal, molecule)

    @classmethod
    def from_symmetry(cls, mol_pg_symbol, sg_name=None, sg_number=None, crys_pg_symbol=None, T=None):
        """Build from bare point-group/space-group symbols, with no coordinates at all.

        Exactly one of `sg_name`, `sg_number`, or `crys_pg_symbol` must be given for the
        crystal; the molecule is given only by its point-group symbol `mol_pg_symbol`.
        """
        crystal = CrystalProjector(sg_name=sg_name, sg_number=sg_number, pg_symbol=crys_pg_symbol)
        molecule = MoleculeProjector(pg_symbol=mol_pg_symbol)
        return cls(crystal, molecule, T=T)
