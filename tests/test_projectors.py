import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from symmscreen import CrystalProjector, MoleculeProjector

# One representative space group per quadrupole survival class, Q_k (paper Table 1).
REPRESENTATIVE_SG = {
    2: 5,  # P-1, triclinic
    10: 3,  # P2/m, monoclinic
    47: 2,  # Pmmm, orthorhombic
    123: 1,  # P4/mmm, tetragonal
    225: 0,  # Fm-3m, cubic
}


@pytest.mark.parametrize("sg_number,expected_class", REPRESENTATIVE_SG.items())
def test_quadrupole_class_matches_paper_table(sg_number, expected_class):
    crystal = CrystalProjector(sg_number=sg_number)
    assert crystal.quadrupole_class == expected_class
    assert crystal.kappa(2) == pytest.approx(expected_class / 5, abs=1e-6)


@pytest.mark.parametrize("sg_number", list(REPRESENTATIVE_SG))
def test_crystal_projector_is_a_projector(sg_number):
    crystal = CrystalProjector(sg_number=sg_number)
    for l in range(4):
        Pi = crystal.matrix(l)
        assert np.allclose(Pi @ Pi, Pi, atol=1e-6)
        rank = np.linalg.matrix_rank(Pi, tol=1e-4)
        assert abs(rank - np.trace(Pi)) < 1e-3


def test_lattice_frame_and_pure_frame_give_same_kappa():
    """kappa is basis-independent: a real lattice shouldn't change it (App. B.2)."""
    crystal_pure = CrystalProjector(sg_number=76)  # P4_1, tetragonal, non-orthogonal-free case
    crystal_lattice = CrystalProjector(
        sg_number=76, a=[5.0, 0.0, 0.0], b=[0.0, 5.0, 0.0], c=[0.0, 0.0, 7.0]
    )
    for l in range(4):
        assert crystal_pure.kappa(l) == pytest.approx(crystal_lattice.kappa(l), abs=1e-6)


def _benzene_atoms(rotation=None, seed=0):
    r = 1.4
    coords = np.array(
        [[r * np.cos(n * np.pi / 3), r * np.sin(n * np.pi / 3), 0.0] for n in range(6)]
    )
    rng = np.random.default_rng(seed)
    coords = coords + rng.normal(scale=0.01, size=coords.shape)  # A touch of "DFT noise".
    if rotation is not None:
        coords = (rotation @ coords.T).T
    return [["C", *c] for c in coords]


def test_molecule_projector_is_a_projector_after_rotation():
    atoms = _benzene_atoms(rotation=Rotation.random(random_state=1).as_matrix())
    molecule = MoleculeProjector(atoms=atoms)
    for l in range(4):
        Pi = molecule.matrix(l)
        assert np.allclose(Pi @ Pi, Pi, atol=1e-3)


def test_molecule_projector_from_pg_symbol_matches_atoms_kappa():
    """An idealised D6h molecule (symbol-only) should have the same kappa_l as
    a numerically-realised D6h benzene ring (atoms-based)."""
    atoms = _benzene_atoms()
    molecule_atoms = MoleculeProjector(atoms=atoms)
    molecule_symbol = MoleculeProjector(pg_symbol="D6h")
    for l in range(4):
        assert molecule_atoms.kappa(l) == pytest.approx(molecule_symbol.kappa(l), abs=1e-2)


def test_molecule_quadrupole_class_matches_paper_table():
    """D6h is hexagonal, Q1 (k=1), same table as the crystal point groups."""
    molecule = MoleculeProjector(pg_symbol="D6h")
    assert molecule.quadrupole_class == 1
    assert molecule.kappa(2) == pytest.approx(1 / 5, abs=1e-6)


def test_molecule_projector_from_cif_forwards_tolerance(synthetic_cif):
    """Real, refined coordinates are rarely exactly symmetric, so `from_cif` needs a
    way to loosen pymatgen's default 0.05 Angstrom tolerance (e.g. caffeine's mirror
    plane is only detected at tolerance >= 0.2)."""
    molecule = MoleculeProjector.from_cif(synthetic_cif, tolerance=0.3)
    assert molecule.tolerance == 0.3


def test_rejects_ambiguous_or_missing_construction():
    with pytest.raises(ValueError):
        CrystalProjector()
    with pytest.raises(ValueError):
        CrystalProjector(sg_number=2, pg_symbol="D2h")
    with pytest.raises(ValueError):
        MoleculeProjector()
    with pytest.raises(ValueError):
        MoleculeProjector(atoms=[["C", 0, 0, 0]], pg_symbol="C1")
