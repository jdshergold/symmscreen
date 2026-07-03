"""Coordinate-aware quadrupole shape vector for the Lambda_coord estimator.

See Sec. "Coordinate-aware estimator" of the paper: the molecule's principal
axes (eigenvectors of the coordinate covariance matrix) are converted into
the l=2 real-spherical-harmonic vector that Lambda_coord is contracted
against.
"""

import numpy as np


def coordinate_principal_axes(atoms):
    """Principal coordinate axes of a centred molecule.

    Returns eigenvalues and eigenvectors of X^T X / N, sorted from largest to
    smallest eigenvalue. Eigenvector signs are fixed deterministically so the
    largest-magnitude Cartesian component is positive.
    """
    coords = np.array([a[1:] for a in atoms], dtype=float)
    coords -= coords.mean(axis=0)
    covariance = coords.T @ coords / len(coords)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    axes = []
    for idx in range(3):
        axis = vectors[:, idx].copy()
        pivot = np.argmax(np.abs(axis))
        if axis[pivot] < 0:
            axis *= -1
        axes.append(axis)
    return values, axes


def quadrupole_shape_vector(axis):
    """Real l=2 quadrupole vector for an undirected coordinate axis.

    Component order is m = (-2, -1, 0, 1, 2), matching the real VSDM/Wigner basis.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("Cannot build a quadrupole shape vector from a zero axis.")
    nx, ny, nz = axis / norm
    vector = np.array(
        [
            np.sqrt(15.0 / (4.0 * np.pi)) * nx * ny,
            np.sqrt(15.0 / (4.0 * np.pi)) * ny * nz,
            np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * nz * nz - 1.0),
            np.sqrt(15.0 / (4.0 * np.pi)) * nx * nz,
            np.sqrt(15.0 / (16.0 * np.pi)) * (nx * nx - ny * ny),
        ]
    )
    return vector / np.linalg.norm(vector)


def coordinate_shape_axis(
    atoms,
    axiality_threshold=0.6,
    transverse_threshold=0.20,
    planarity_threshold=0.30,
):
    """Coordinate-aware choice of the important principal axis.

    Axial molecules (long along one axis) use the longest principal axis;
    planar molecules use the shortest (out-of-plane) axis; anything in
    between uses the element-wise median direction of the three axes.
    """
    values, axes = coordinate_principal_axes(atoms)
    if values[0] < 1e-12:
        return axes[0]

    axiality = 1.0 - values[1] / values[0]
    planarity = (values[1] - values[2]) / values[0]
    if axiality >= axiality_threshold and planarity <= transverse_threshold:
        return axes[0]
    if planarity >= planarity_threshold:
        return axes[2]

    median_axis = np.median(np.array(axes), axis=0)
    if np.linalg.norm(median_axis) < 1e-12:
        median_axis = axes[1]
    return median_axis / np.linalg.norm(median_axis)


def coordinate_shape_vector(atoms, **kwargs):
    """Coordinate-aware normalised l=2 shape vector for a molecule's atoms."""
    return quadrupole_shape_vector(coordinate_shape_axis(atoms, **kwargs))
