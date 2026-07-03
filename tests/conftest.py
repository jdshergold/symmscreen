import numpy as np
import pytest
from pymatgen.core import Lattice, Structure


@pytest.fixture
def synthetic_cif(tmp_path):
    """A small, hand-built (non-CCDC) synthetic molecular crystal for testing.

    Three well-separated "C" atoms forming a scalene (C1-symmetric) triangle, placed
    at a general position in space group P-1 (No. 2, triclinic), so `from_cif` has a
    real space group + lattice + molecule to parse without depending on any
    externally-sourced crystallographic data.
    """
    lattice = Lattice.orthorhombic(10.0, 11.0, 12.0)
    species = ["C", "C", "C"]
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.10 + 1.5 / 10.0, 0.10, 0.10],
            [0.10 + 0.7 / 10.0, 0.10 + 1.3 / 11.0, 0.10],
        ]
    )
    structure = Structure.from_spacegroup(2, lattice, species, coords)
    path = tmp_path / "synthetic.cif"
    structure.to(filename=str(path), fmt="cif")
    return path
