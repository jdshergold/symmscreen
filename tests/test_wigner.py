import numpy as np
from scipy.spatial.transform import Rotation

from symmscreen._wigner import wigner_g


def test_identity_rotation_gives_identity_matrix():
    for l in range(5):
        G = wigner_g(l, np.eye(3))
        assert np.allclose(G, np.eye(2 * l + 1), atol=1e-8)


def test_wigner_g_is_orthogonal():
    R = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
    for l in range(5):
        G = wigner_g(l, R)
        assert np.allclose(G @ G.T, np.eye(2 * l + 1), atol=1e-8)


def test_wigner_g_is_a_homomorphism():
    """G(R1 R2) = G(R1) G(R2): the defining property of a group representation."""
    R1 = Rotation.from_euler("xyz", [12, 77, 143], degrees=True).as_matrix()
    R2 = Rotation.from_euler("zyx", [200, 10, 305], degrees=True).as_matrix()
    for l in range(5):
        G1 = wigner_g(l, R1)
        G2 = wigner_g(l, R2)
        G12 = wigner_g(l, R1 @ R2)
        assert np.allclose(G1 @ G2, G12, atol=1e-6)
