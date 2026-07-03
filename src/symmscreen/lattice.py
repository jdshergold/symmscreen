"""Lattice matrices for expressing point-group operations in Cartesian coordinates."""

import numpy as np

from .point_groups import pg_ops


def pure_metric(sg_name=None, sg_number=None, pg_symbol=None):
    """Pure crystal metric for a point group: M = (1/N) sum_i R_i^T R_i."""
    ops = pg_ops(sg_name=sg_name, sg_number=sg_number, pg_symbol=pg_symbol)
    return sum(op.T @ op for op in ops) / len(ops)


def pure_lattice_matrix(sg_name=None, sg_number=None, pg_symbol=None):
    """Lattice matrix A (columns = lattice vectors), A = cholesky(pure_metric).T.

    This gives a Cartesian frame determined purely by the point-group symmetry,
    with no reference to any specific crystal geometry.
    """
    M = pure_metric(sg_name=sg_name, sg_number=sg_number, pg_symbol=pg_symbol)
    return np.linalg.cholesky(M).T


def crystal_lattice_matrix(a, b, c):
    """Lattice matrix A whose columns are the real lattice vectors a, b, c."""
    return np.column_stack([a, b, c])
