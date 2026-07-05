"""Extract space group, lattice, and molecule coordinates from a CIF file."""

import warnings

from pymatgen.analysis.dimensionality import get_structure_components
from pymatgen.analysis.local_env import JmolNN
from pymatgen.io.cif import CifParser


def _parse_structure(cif_path):
    for tol in (1.0, 2.1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structures = CifParser(cif_path, occupancy_tolerance=tol).parse_structures(
                    primitive=False
                )
            if structures:
                return structures[0]
        except ValueError:
            continue
    raise ValueError(f"Could not parse CIF: {cif_path}")


def _all_molecules(structure):
    """Every connected molecule's atoms in a parsed pymatgen Structure (already includes
    all symmetry-equivalent copies, since pymatgen expands the full cell on parsing)."""
    nn = JmolNN()
    bond_structure = nn.get_bonded_structure(structure)
    components = get_structure_components(bond_structure, inc_molecule_graph=True)

    molecules = []
    for component in components:
        mol_graph = component["molecule_graph"]
        atoms = []
        for site in mol_graph.molecule.sites:
            element = list(site.species.as_dict().keys())[0]
            atoms.append([element, float(site.coords[0]), float(site.coords[1]), float(site.coords[2])])
        molecules.append(atoms)
    return molecules


def cif_crystal_info(cif_path, molecule_index=0):
    """Extract the space group, lattice vectors, and one molecule's atoms from a CIF file.

    The molecule is picked by `molecule_index` from the list of connected components
    returned by pymatgen. For ordered crystals all molecules are equivalent, so the
    default of 0 is fine. For disordered structures, filter the structure manually
    before calling this.

    Args:
        cif_path: Path to the CIF file.
        molecule_index: Which connected-component molecule to use (default 0).

    Returns:
        sg_name: Hermann-Mauguin space-group symbol (str).
        a, b, c: Lattice vectors as 1-D numpy arrays (Angstrom).
        atoms: List of [symbol, x, y, z] for every atom in the chosen molecule.
    """
    structure = _parse_structure(cif_path)
    sg_name = structure.get_space_group_info()[0]
    lat = structure.lattice.matrix  # Rows are a, b, c.
    a, b, c = lat[0], lat[1], lat[2]

    molecules = _all_molecules(structure)
    if not molecules:
        raise ValueError(f"No molecules found in CIF: {cif_path}")
    if molecule_index >= len(molecules):
        raise IndexError(
            f"molecule_index {molecule_index} out of range; "
            f"CIF contains {len(molecules)} molecule(s)."
        )

    return sg_name, a, b, c, molecules[molecule_index]


def all_molecules_from_cif(cif_path):
    """Extract the lattice vectors and every molecule's atoms from a CIF file.

    Unlike `cif_crystal_info`, which returns one representative molecule, this returns
    every molecular component in the (symmetry-expanded) cell. Used for visualising the
    full unit-cell packing.

    Returns:
        a, b, c: Lattice vectors as 1-D numpy arrays (Angstrom).
        molecules: List of molecules, each a list of [symbol, x, y, z].
    """
    structure = _parse_structure(cif_path)
    lat = structure.lattice.matrix
    a, b, c = lat[0], lat[1], lat[2]
    return a, b, c, _all_molecules(structure)
