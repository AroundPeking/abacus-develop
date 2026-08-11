from __future__ import annotations

import math
import unittest

import numpy as np

from exx_thc.metrics import metric_relative_frobenius, whiten


class CoulombMetricTest(unittest.TestCase):
    def test_whitening_preserves_full_coulomb_metric_contraction(self):
        rng = np.random.default_rng(602)
        trial = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        unitary, _ = np.linalg.qr(trial)
        eigenvalues = np.asarray([4.0, 0.7, 0.0])
        metric = (unitary * eigenvalues) @ unitary.conj().T
        cbar = rng.normal(size=(3, 2, 4)) + 1j * rng.normal(size=(3, 2, 4))

        result = whiten(metric, cbar)

        self.assertEqual(result.tensor.shape, (2, 2, 4))
        self.assertEqual(result.transform.shape, (2, 3))
        self.assertEqual(result.tensor.dtype, np.complex128)
        self.assertEqual(result.removed_zero_modes, 1)
        self.assertLess(result.hermiticity_rel, 1.0e-14)
        np.testing.assert_allclose(result.eigenvalues, np.sort(eigenvalues), atol=1.0e-13)
        whitened_norm_squared = float(np.vdot(result.tensor, result.tensor).real)
        direct_metric_norm_squared = float(
            np.einsum("miv,mn,niv->", cbar.conj(), metric, cbar).real
        )
        self.assertAlmostEqual(whitened_norm_squared, direct_metric_norm_squared, places=11)
        np.testing.assert_allclose(
            result.tensor,
            np.einsum("xm,miv->xiv", result.transform, cbar),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_records_residual_before_hermitizing_full_matrix(self):
        metric = np.asarray(
            [[2.0, 0.3 + 0.2j], [0.3 - 0.1j, 1.0]], dtype=np.complex128
        )
        cbar = np.ones((2, 1, 1), dtype=np.complex128)
        expected = np.linalg.norm(metric - metric.conj().T) / np.linalg.norm(metric)

        result = whiten(metric, cbar)

        self.assertAlmostEqual(result.hermiticity_rel, expected, places=15)
        hermitized = 0.5 * metric + 0.5 * metric.conj().T
        expected_norm_squared = float(
            np.einsum("miv,mn,niv->", cbar.conj(), hermitized, cbar).real
        )
        self.assertAlmostEqual(
            float(np.vdot(result.tensor, result.tensor).real), expected_norm_squared, places=13
        )

    def test_small_scale_negative_eigenvalue_uses_relative_psd_gate(self):
        metric = np.diag([1.0e-6, -5.0e-11]).astype(np.complex128)
        cbar = np.ones((2, 1, 1), dtype=np.complex128)

        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            whiten(metric, cbar)

    def test_clips_threshold_scale_negative_perturbation_and_removes_zero_mode(self):
        metric = np.diag([1.0, -5.0e-11]).astype(np.complex128)
        cbar = np.asarray([[[2.0 + 1.0j]], [[7.0 - 3.0j]]], dtype=np.complex128)

        result = whiten(metric, cbar)

        self.assertEqual(result.tensor.shape, (1, 1, 1))
        self.assertEqual(result.transform.shape, (1, 2))
        self.assertEqual(result.removed_zero_modes, 1)
        np.testing.assert_array_equal(result.eigenvalues, np.asarray([0.0, 1.0]))
        np.testing.assert_allclose(np.abs(result.tensor[0]), np.abs(cbar[0]), atol=1.0e-14)

    def test_zero_metric_returns_zero_auxiliary_whitened_tensor(self):
        metric = np.zeros((3, 3), dtype=np.complex128)
        cbar = np.ones((3, 2, 4), dtype=np.complex128)

        result = whiten(metric, cbar)

        self.assertEqual(result.tensor.shape, (0, 2, 4))
        self.assertEqual(result.transform.shape, (0, 3))
        self.assertEqual(result.removed_zero_modes, 3)
        self.assertEqual(result.hermiticity_rel, 0.0)
        np.testing.assert_array_equal(result.eigenvalues, np.zeros(3))

    def test_pure_antihermitian_metric_uses_hermitized_zero_for_psd_gate(self):
        metric = np.asarray([[0.0, 2.0j], [2.0j, 0.0]], dtype=np.complex128)
        cbar = np.ones((2, 1, 3), dtype=np.complex128)

        result = whiten(metric, cbar)

        self.assertGreater(result.hermiticity_rel, 0.0)
        self.assertEqual(result.tensor.shape, (0, 1, 3))
        self.assertEqual(result.transform.shape, (0, 2))
        self.assertEqual(result.removed_zero_modes, 2)
        np.testing.assert_array_equal(result.eigenvalues, np.zeros(2))

    def test_whiten_rejects_shape_dtype_nonfinite_and_zero_scale_negative(self):
        complex_cbar = np.ones((2, 1, 1), dtype=np.complex128)
        cases = [
            (np.ones((2, 3), dtype=np.complex128), complex_cbar),
            (np.eye(2, dtype=np.complex128), np.ones((2, 2), dtype=np.complex128)),
            (np.eye(2, dtype=np.complex128), np.ones((3, 1, 1), dtype=np.complex128)),
            (np.eye(2), complex_cbar),
            (np.eye(2, dtype=np.complex128), np.ones((2, 1, 1))),
            (np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.complex128), complex_cbar),
            (
                np.eye(2, dtype=np.complex128),
                np.asarray([[[1.0]], [[np.inf]]], dtype=np.complex128),
            ),
            (np.diag([0.0, -1.0e-30]).astype(np.complex128), complex_cbar),
        ]
        for metric, cbar in cases:
            with self.subTest(metric_shape=metric.shape, cbar_shape=cbar.shape), self.assertRaises(
                ValueError
            ):
                whiten(metric, cbar)

    def test_active_relative_to_raw_metric_difference_and_zero_boundary(self):
        raw = np.asarray([[2.0, 0.5j], [-0.5j, 1.0]], dtype=np.complex128)
        active = raw.copy()
        active[0, 1] = active[1, 0] = 0.0
        expected = np.linalg.norm(active - raw) / np.linalg.norm(raw)

        self.assertAlmostEqual(metric_relative_frobenius(active, raw), expected, places=15)
        self.assertEqual(
            metric_relative_frobenius(
                np.zeros((2, 2), dtype=np.complex128),
                np.zeros((2, 2), dtype=np.complex128),
            ),
            0.0,
        )
        self.assertTrue(
            math.isinf(
                metric_relative_frobenius(
                    np.eye(2, dtype=np.complex128),
                    np.zeros((2, 2), dtype=np.complex128),
                )
            )
        )

    def test_metric_difference_rejects_shape_dtype_and_nonfinite(self):
        cases = [
            (np.eye(2, dtype=np.complex128), np.eye(3, dtype=np.complex128)),
            (np.eye(2), np.eye(2, dtype=np.complex128)),
            (
                np.asarray([[1.0, np.inf], [0.0, 1.0]], dtype=np.complex128),
                np.eye(2, dtype=np.complex128),
            ),
        ]
        for active, raw in cases:
            with self.subTest(active_shape=active.shape, raw_shape=raw.shape), self.assertRaises(
                ValueError
            ):
                metric_relative_frobenius(active, raw)


if __name__ == "__main__":
    unittest.main()
