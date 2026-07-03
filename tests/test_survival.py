import numpy as np
import pytest

from symmscreen import CombinedSurvival, CrystalProjector


def test_benzene_d6h_in_d2h_matches_paper_agnostic_estimator():
    """Eq. (differentGroupAgnostic): for non-cubic L and H, <Lambda^(L)>_T ~ sqrt(kappa_2^(L)).
    D2h is in the orthorhombic Q2 class, so kappa_2^(L) = 2/5."""
    survival = CombinedSurvival.from_symmetry(mol_pg_symbol="D6h", crys_pg_symbol="D2h")
    assert survival.lambda_L_avg(l_max=12) == pytest.approx(np.sqrt(2 / 5), rel=0.05)


def test_benzene_d6h_in_d2h_matches_paper_ideal_headline_number():
    """paper_draft.tex ~line 1507: for a benzene-like D6h molecule embedded in a D2h
    crystal, the *ideal* (cross-molecule-ranking) estimator <Lambda>_T ~ sqrt(2)/5 ~ 0.282."""
    survival = CombinedSurvival.from_symmetry(mol_pg_symbol="D6h", crys_pg_symbol="D2h")
    assert survival.lambda_ideal_avg(l_max=12) == pytest.approx(np.sqrt(2) / 5, rel=0.05)


def test_xi_inf_bounds_for_benzene_example():
    """Sec. 'Combined anisotropy survival': 1/48 <= xi_inf(T) <= 1/12 for D6h in D2h."""
    survival = CombinedSurvival.from_symmetry(mol_pg_symbol="D6h", crys_pg_symbol="D2h")
    xi_upper = min(survival.crystal.xi_inf, survival.molecule.xi_inf)
    xi_lower = survival.crystal.xi_inf * survival.molecule.xi_inf
    assert xi_lower == pytest.approx(1 / 48, rel=1e-6)
    assert xi_upper == pytest.approx(1 / 12, rel=1e-6)


def test_pure_crystal_only_reduces_to_crystal_projector():
    crystal = CrystalProjector(pg_symbol="D2h")
    survival = CombinedSurvival(crystal)
    assert np.allclose(survival.matrix(2), crystal.matrix(2))
    with pytest.raises(ValueError):
        survival.lambda_L()
    with pytest.raises(ValueError):
        survival.lambda_L_avg()
    with pytest.raises(ValueError):
        survival.lambda_ideal_avg()
    with pytest.raises(ValueError):
        survival.lambda_coord()


def test_orientation_specific_lambda_L_bounded_by_zero_and_one():
    rng = np.random.default_rng(0)
    for _ in range(5):
        from scipy.spatial.transform import Rotation

        T = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        survival = CombinedSurvival.from_symmetry(mol_pg_symbol="C1", crys_pg_symbol="D2h", T=T)
        value = survival.lambda_L(l_max=8)
        assert 0.0 <= value <= 1.0 + 1e-8


def test_from_cif_end_to_end(synthetic_cif):
    survival = CombinedSurvival.from_cif(synthetic_cif)
    value = survival.lambda_coord()
    assert 0.0 < value <= 1.0 + 1e-8
    assert survival.crystal.quadrupole_class in {0, 1, 2, 3, 5}
    assert survival.molecule.quadrupole_class in {0, 1, 2, 3, 5}


def test_cif_convenience_functions_match_combined_survival(synthetic_cif):
    import symmscreen as ss

    survival = CombinedSurvival.from_cif(synthetic_cif)
    assert ss.crystal_quadrupole_class(synthetic_cif) == survival.crystal.quadrupole_class
    assert ss.molecule_quadrupole_class(synthetic_cif) == survival.molecule.quadrupole_class
    assert ss.lambda_coord(synthetic_cif) == pytest.approx(survival.lambda_coord())


def test_combined_survival_from_cif_forwards_mol_tolerance(synthetic_cif):
    survival = CombinedSurvival.from_cif(synthetic_cif, mol_tolerance=0.3)
    assert survival.molecule.tolerance == 0.3


def test_symmetry_convenience_functions_match_combined_survival():
    import symmscreen as ss

    survival = CombinedSurvival.from_symmetry(mol_pg_symbol="D6h", crys_pg_symbol="D2h")
    assert ss.lambda_L_avg("D6h", crys_pg_symbol="D2h") == pytest.approx(survival.lambda_L_avg())
    assert ss.lambda_ideal_avg("D6h", crys_pg_symbol="D2h") == pytest.approx(survival.lambda_ideal_avg())
