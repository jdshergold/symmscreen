"""symmscreen: symmetry projectors and modulation-loss (Lambda) estimators for screening
molecular crystals for directional dark matter detection.

Companion package to "The Role of Symmetries in Dark Matter Detector Design"
(Lillard, Shergold, Smirnov). See the paper for the physics; this package implements
only the symmetry-projector and Lambda-estimator machinery (Secs. 3-6), so a crystal
can be screened from a CIF, or from bare space-group/point-group symbols, without any
electronic-structure calculation.
"""

from .cif import cif_crystal_info
from .point_groups import (
    HM_TO_SCHOENFLIES,
    MOLECULE_SYMMETRY_TOLERANCE,
    SCHOENFLIES_TO_HM,
    pg_ops,
)
from .projectors import CrystalProjector, MoleculeProjector
from .screening import (
    crystal_quadrupole_class,
    lambda_L_avg,
    lambda_coord,
    lambda_ideal_avg,
    molecule_quadrupole_class,
)
from .survival import CombinedSurvival

__all__ = [
    "CombinedSurvival",
    "CrystalProjector",
    "HM_TO_SCHOENFLIES",
    "MOLECULE_SYMMETRY_TOLERANCE",
    "MoleculeProjector",
    "SCHOENFLIES_TO_HM",
    "cif_crystal_info",
    "crystal_quadrupole_class",
    "lambda_L_avg",
    "lambda_coord",
    "lambda_ideal_avg",
    "molecule_quadrupole_class",
    "pg_ops",
]
