"""Point-group operations, as 3x3 matrices, from space groups, point groups, or atoms."""

import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.symmetry.groups import PointGroup, SpaceGroup

from ._clean import clean_matrix

# Matches pymatgen's own PointGroupAnalyzer default. Real, refined crystallographic
# coordinates are rarely exactly symmetric, so a stricter (smaller) tolerance can fail
# to detect symmetry a chemist would consider obvious (e.g. caffeine's mirror plane
# needs tolerance >= 0.2). Lower it if you instead want to require closer-to-exact
# symmetry, e.g. to avoid accepting a merely approximately-symmetric structure as exact.
MOLECULE_SYMMETRY_TOLERANCE = 0.3

# ==== Bidirectional symbol lookup for the 32 crystallographic point groups. ==== #
HM_TO_SCHOENFLIES = {
    "1": "C1",
    "-1": "Ci",
    "2": "C2",
    "m": "Cs",
    "2/m": "C2h",
    "222": "D2",
    "mm2": "C2v",
    "mmm": "D2h",
    "4": "C4",
    "-4": "S4",
    "4/m": "C4h",
    "422": "D4",
    "4mm": "C4v",
    "-42m": "D2d",
    "4/mmm": "D4h",
    "3": "C3",
    "-3": "C3i",
    "32": "D3",
    "3m": "C3v",
    "-3m": "D3d",
    "6": "C6",
    "-6": "C3h",
    "6/m": "C6h",
    "622": "D6",
    "6mm": "C6v",
    "-6m2": "D3h",
    "6/mmm": "D6h",
    "23": "T",
    "m-3": "Th",
    "432": "O",
    "-43m": "Td",
    "m-3m": "Oh",
}

SCHOENFLIES_TO_HM = {v: k for k, v in HM_TO_SCHOENFLIES.items()}


def pg_ops_from_pg_symbol(pg_symbol):
    """Point group operations as 3x3 matrices from a point group symbol.

    Both Schoenflies and Hermann-Mauguin symbols are accepted.
    """
    hm_symbol = SCHOENFLIES_TO_HM.get(pg_symbol, pg_symbol)
    pg = PointGroup(hm_symbol)
    return [op.rotation_matrix for op in pg.symmetry_ops]


def pg_ops_from_sg_symbol(sg_symbol):
    """Point group operations as 3x3 matrices from a Hermann-Mauguin space group symbol."""
    pg = PointGroup.from_space_group(sg_symbol)
    return [op.rotation_matrix for op in pg.symmetry_ops]


def pg_ops_from_sg_number(sg_number):
    """Point group operations as 3x3 matrices from a space group number (1-230)."""
    assert 1 <= sg_number <= 230, "Space group number must be between 1 and 230."
    sg = SpaceGroup.from_int_number(sg_number)
    pg = PointGroup.from_space_group(sg.symbol)
    return [op.rotation_matrix for op in pg.symmetry_ops]


def pg_ops_from_atoms(atoms, tolerance=MOLECULE_SYMMETRY_TOLERANCE):
    """Point group operations as 3x3 matrices from a list of [species, x, y, z] atoms.

    Operations are returned in the frame of the centred molecular coordinates.
    """
    species = [a[0] for a in atoms]
    coords = np.array([a[1:] for a in atoms], dtype=float)
    coords -= coords.mean(axis=0)  # Centre; translations don't matter for point groups.
    mol = Molecule(species, coords)
    pga = PointGroupAnalyzer(mol, tolerance=tolerance)
    return [clean_matrix(op.rotation_matrix) for op in pga.get_symmetry_operations()]


def pg_ops(sg_name=None, sg_number=None, pg_symbol=None, atoms=None, tolerance=MOLECULE_SYMMETRY_TOLERANCE):
    """Point group operations as 3x3 matrices from a space group, point group, or atoms.

    Exactly one of `sg_name`, `sg_number`, `pg_symbol`, or `atoms` should be given.
    The space group name must be in Hermann-Mauguin notation. The point group symbol
    can be either Hermann-Mauguin or Schoenflies.
    """
    if pg_symbol is not None:
        return pg_ops_from_pg_symbol(pg_symbol)
    elif sg_number is not None:
        return pg_ops_from_sg_number(sg_number)
    elif sg_name is not None:
        return pg_ops_from_sg_symbol(sg_name)
    elif atoms is not None:
        return pg_ops_from_atoms(atoms, tolerance=tolerance)
    else:
        raise ValueError(
            "Must provide one of space group name, space group number, point group symbol, or atoms."
        )
