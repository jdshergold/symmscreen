"""One-line convenience functions for the two most common inputs: a CIF, or bare symmetry labels.

For quickly sweeping many crystal structures with known coordinates, e.g.:

    import symmscreen as ss

    for path in cif_dir.glob("*.cif"):
        print(path.name, ss.lambda_coord(path))

`lambda_L_avg` and `lambda_ideal_avg` average over the (generally unknown) molecular
orientation, so they take symmetry labels rather than a CIF -- see
`CombinedSurvival.from_symmetry` for what each argument means.

If you want more than one metric for the same crystal, build a `CombinedSurvival` directly
(`CombinedSurvival.from_cif(path)` or `CombinedSurvival.from_symmetry(...)`) and call the
methods you need on it -- it caches the crystal/molecule projectors internally, so nothing
is recomputed between calls.
"""

from .survival import CombinedSurvival


def crystal_quadrupole_class(cif_path):
    """k = rank(Pi_2^(L)) in {0, 1, 2, 3, 5}, the crystal's quadrupole survival class Q_k."""
    return CombinedSurvival.from_cif(cif_path).crystal.quadrupole_class


def molecule_quadrupole_class(cif_path, molecule_index=0, mol_tolerance=None):
    """k = rank(Pi_2^(H)) in {0, 1, 2, 3, 5}, the molecule's own quadrupole survival class Q_k.

    `mol_tolerance` is passed to pymatgen's `PointGroupAnalyzer` (default 0.3 Angstrom,
    pymatgen's own default, if omitted); lower it for stricter symmetry detection.
    """
    return CombinedSurvival.from_cif(
        cif_path, molecule_index=molecule_index, mol_tolerance=mol_tolerance
    ).molecule.quadrupole_class


def lambda_L_avg(mol_pg_symbol, sg_name=None, sg_number=None, crys_pg_symbol=None, l_max=12):
    """<Lambda^(L)>_T for a molecule/crystal given purely by symmetry labels."""
    survival = CombinedSurvival.from_symmetry(
        mol_pg_symbol=mol_pg_symbol, sg_name=sg_name, sg_number=sg_number, crys_pg_symbol=crys_pg_symbol
    )
    return survival.lambda_L_avg(l_max=l_max)


def lambda_ideal_avg(mol_pg_symbol, sg_name=None, sg_number=None, crys_pg_symbol=None, l_max=12):
    """<Lambda>_T, the absolute cross-molecule ranking estimator, from symmetry labels alone."""
    survival = CombinedSurvival.from_symmetry(
        mol_pg_symbol=mol_pg_symbol, sg_name=sg_name, sg_number=sg_number, crys_pg_symbol=crys_pg_symbol
    )
    return survival.lambda_ideal_avg(l_max=l_max)


def lambda_coord(cif_path, molecule_index=0, mol_tolerance=None):
    """Lambda_coord for the crystal/molecule in this CIF.

    `mol_tolerance` is passed to pymatgen's `PointGroupAnalyzer` (default 0.3 Angstrom,
    pymatgen's own default, if omitted); lower it for stricter symmetry detection.
    """
    return CombinedSurvival.from_cif(
        cif_path, molecule_index=molecule_index, mol_tolerance=mol_tolerance
    ).lambda_coord()
