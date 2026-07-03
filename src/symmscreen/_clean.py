"""Numerical cleanup helpers."""

import numpy as np


def clean_matrix(matrix, tol=1e-5):
    """Zero out entries of `matrix` with absolute value below `tol`."""
    cleaned = matrix.copy()
    cleaned[np.abs(cleaned) < tol] = 0
    return cleaned
