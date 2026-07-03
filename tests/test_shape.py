import numpy as np

from symmscreen.shape import coordinate_shape_axis, quadrupole_shape_vector


def test_quadrupole_shape_vector_is_normalised():
    v = quadrupole_shape_vector(np.array([1.0, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_axial_molecule_uses_longest_axis():
    # A long, thin "molecule" of atoms strung out along x.
    atoms = [["C", x, 0.0, 0.0] for x in np.linspace(-2, 2, 9)]
    axis = coordinate_shape_axis(atoms)
    assert np.isclose(abs(axis[0]), 1.0, atol=1e-6)


def test_planar_molecule_uses_out_of_plane_axis():
    # A flat, disc-shaped ring of atoms in the x-y plane.
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    atoms = [["C", np.cos(t), np.sin(t), 0.0] for t in theta]
    axis = coordinate_shape_axis(atoms)
    assert np.isclose(abs(axis[2]), 1.0, atol=1e-6)
