from __future__ import annotations

import unittest

import numpy as np

from exx_thc.occupied import occupied_factor


class OccupiedFactorTest(unittest.TestCase):
    def test_preserves_density_contraction_in_occupied_space(self) -> None:
        rng = np.random.default_rng(1147)
        trial = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
        q, _ = np.linalg.qr(trial)
        occupations = np.array([1.0, 0.7, 0.2])
        density = (q * occupations) @ q.conj().T
        c = rng.normal(size=(5, 6)) + 1j * rng.normal(size=(5, 6))

        factor = occupied_factor(density, eigenvalue_tol=0.0)
        cbar = c @ factor.O
        c_occ = cbar @ factor.O_pinv

        self.assertEqual(factor.O.shape, (6, 3))
        self.assertLess(np.linalg.norm(factor.O_pinv), 10.0)
        np.testing.assert_allclose(
            factor.O @ factor.O.conj().T, density, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            c_occ @ density @ c_occ.conj().T,
            c @ density @ c.conj().T,
            rtol=1e-12,
            atol=1e-12,
        )
        projector = factor.O @ factor.O_pinv
        np.testing.assert_allclose(projector, projector.conj().T, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(projector @ projector, projector, rtol=1e-12, atol=1e-12)
        self.assertLess(abs(factor.discarded_trace), 1e-14)

    def test_zero_density_has_empty_factors(self) -> None:
        factor = occupied_factor(np.zeros((4, 4), dtype=np.complex128), 0.0)

        self.assertEqual(factor.O.shape, (4, 0))
        self.assertEqual(factor.O_pinv.shape, (0, 4))
        np.testing.assert_array_equal(factor.eigenvalues, np.zeros(4))
        self.assertEqual(factor.discarded_trace, 0.0)

    def test_positive_tolerance_reports_discarded_trace(self) -> None:
        density = np.diag([1.0, 1.0e-5, 0.0]).astype(np.complex128)

        factor = occupied_factor(density, 1.0e-4)

        self.assertEqual(factor.O.shape, (3, 1))
        self.assertAlmostEqual(factor.discarded_trace, 1.0e-5, places=15)
        np.testing.assert_allclose(
            factor.O @ factor.O.conj().T,
            np.diag([1.0, 0.0, 0.0]),
            rtol=0.0,
            atol=1e-14,
        )

    def test_rejects_non_psd_non_square_nonfinite_and_bad_tolerance(self) -> None:
        cases = [
            (np.diag([1.0, -1.0e-3]), 0.0),
            (np.ones((2, 3)), 0.0),
            (np.array([[1.0, np.nan], [0.0, 1.0]]), 0.0),
            (np.eye(2), -1.0),
            (np.eye(2), np.inf),
        ]
        for density, tolerance in cases:
            with self.subTest(shape=density.shape, tolerance=tolerance), self.assertRaises(
                ValueError
            ):
                occupied_factor(density, tolerance)


if __name__ == "__main__":
    unittest.main()
