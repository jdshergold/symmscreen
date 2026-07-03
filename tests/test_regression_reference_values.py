"""Regression test locking symmscreen's numbers to the original research implementation.

These reference values were produced once by running the same (seed=42, 5-atom,
`scale=1.5` Gaussian) synthetic molecule through the pre-extraction research code in
`../src/symmetry/projectors.py` + `../src/rate/modulation.py` (CombinedSurvival +
Modulation.proxy / proxy_rms_avg / proxy_abs_rms_avg / coordinate_shape_proxy), across
one representative space group per crystal system. symmscreen reproduced them bit-for-bit
at the time this test was written; this pins that fidelity going forward.
"""

import numpy as np
import pytest

from symmscreen import CombinedSurvival, CrystalProjector, MoleculeProjector

REFERENCE = {
    # sg_number: (lambda_L, lambda_L_avg, lambda_ideal_avg, lambda_coord)
    2: (1.000000, 1.000000, 1.000000, 1.000000),
    14: (0.774563, 0.774578, 0.774578, 0.560828),
    61: (0.632394, 0.632422, 0.632422, 0.475431),
    88: (0.447386, 0.447309, 0.447309, 0.172885),
    148: (0.447386, 0.447310, 0.447310, 0.172885),
    167: (0.447242, 0.447230, 0.447230, 0.172885),
    194: (0.447099, 0.447150, 0.447150, 0.172885),
    227: (0.011343, 0.008455, 0.008455, 0.000000),
}


@pytest.fixture(scope="module")
def molecule():
    rng = np.random.default_rng(42)
    atoms = [["C", *rng.normal(scale=1.5, size=3)] for _ in range(5)]
    return MoleculeProjector(atoms=atoms)


@pytest.mark.parametrize("sg_number,expected", REFERENCE.items())
def test_matches_original_implementation(molecule, sg_number, expected):
    crystal = CrystalProjector(sg_number=sg_number)
    survival = CombinedSurvival(crystal, molecule, T=np.eye(3))

    lam_L = survival.lambda_L(l_max=12)
    lam_L_avg = survival.lambda_L_avg(l_max=12)
    lam_ideal_avg = survival.lambda_ideal_avg(l_max=12)
    lam_coord = survival.lambda_coord()

    assert (lam_L, lam_L_avg, lam_ideal_avg, lam_coord) == pytest.approx(expected, abs=1e-5)
