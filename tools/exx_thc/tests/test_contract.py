from __future__ import annotations

import unittest

import numpy as np

from exx_thc.contract import ExchangeDiagnostics, dense_exchange, thc_exchange


def _case():
    rng = np.random.default_rng(11)
    t = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    x = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    y = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    cbar = np.einsum("mr,ir,vr->miv", t.conj(), x, y)
    raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    metric = raw @ raw.conj().T + 0.5 * np.eye(7)
    return t, x, y, cbar, metric


class ExchangeContractionTest(unittest.TestCase):
    def test_complex_thc_formula_matches_dense_exchange(self):
        t, x, y, cbar, metric = _case()
        self.assertGreater(np.max(np.abs(cbar.imag)), 1.0e-3)

        dense = dense_exchange(cbar, metric)
        thc = thc_exchange(t, x, y, metric)

        np.testing.assert_allclose(thc, dense, rtol=1.0e-11, atol=1.0e-11)
        self.assertEqual(dense.dtype, np.complex128)
        self.assertEqual(thc.dtype, np.complex128)
        self.assertTrue(dense.flags.c_contiguous)
        self.assertTrue(thc.flags.c_contiguous)

    def test_thc_diagnostics_list_only_rank_sized_temporaries(self):
        t, x, y, _, metric = _case()

        result = thc_exchange(t, x, y, metric, diagnostics=True)

        self.assertIsInstance(result, ExchangeDiagnostics)
        shapes = result.temporary_shapes
        self.assertEqual(shapes["vt"], (7, 3))
        self.assertEqual(shapes["z"], (3, 3))
        self.assertEqual(shapes["s"], (3, 3))
        self.assertEqual(shapes["core"], (3, 3))
        self.assertEqual(shapes["xcore"], (5, 3))
        self.assertEqual(shapes["K"], (5, 5))
        self.assertNotIn((7, 5, 4), shapes.values())

    def test_nonhermitian_metric_is_diagnosed_without_symmetrizing_output(self):
        t, x, y, cbar, metric = _case()
        metric = metric.copy()
        metric[0, 1] += 0.7 + 0.4j
        expected = -np.einsum("miv,mn,nkv->ik", cbar, metric, cbar.conj(), optimize=True)

        dense = dense_exchange(cbar, metric, diagnostics=True)
        thc = thc_exchange(t, x, y, metric, diagnostics=True)

        self.assertGreater(dense.hermiticity_rel, 0.0)
        self.assertGreater(thc.hermiticity_rel, 0.0)
        np.testing.assert_allclose(dense.matrix, expected, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(thc.matrix, expected, rtol=1.0e-11, atol=1.0e-11)
        self.assertGreater(np.linalg.norm(dense.matrix - dense.matrix.conj().T), 0.0)

    def test_inputs_are_promoted_and_nonfinite_or_bad_shapes_are_rejected(self):
        dense = dense_exchange(np.ones((2, 1, 1)), np.eye(2))
        thc = thc_exchange(np.ones((2, 1)), np.ones((1, 1)), np.ones((1, 1)), np.eye(2))
        self.assertEqual(dense.dtype, np.complex128)
        self.assertEqual(thc.dtype, np.complex128)

        dense_cases = [
            (np.ones((2, 2)), np.eye(2)),
            (np.ones((2, 1, 1)), np.eye(3)),
            (np.full((2, 1, 1), np.nan), np.eye(2)),
            (np.ones((2, 1, 1)), np.asarray([[1.0, np.inf], [0.0, 1.0]])),
        ]
        for cbar, metric in dense_cases:
            with self.subTest(kind="dense", cbar_shape=cbar.shape), self.assertRaises(ValueError):
                dense_exchange(cbar, metric)

        thc_cases = [
            (np.ones((2, 1, 1)), np.ones((1, 1)), np.ones((1, 1)), np.eye(2)),
            (np.ones((2, 2)), np.ones((1, 1)), np.ones((1, 1)), np.eye(2)),
            (np.ones((2, 1)), np.ones((1, 2)), np.ones((1, 1)), np.eye(2)),
            (np.full((2, 1), np.nan), np.ones((1, 1)), np.ones((1, 1)), np.eye(2)),
        ]
        for t, x, y, metric in thc_cases:
            with self.subTest(kind="thc", t_shape=t.shape), self.assertRaises(ValueError):
                thc_exchange(t, x, y, metric)

    def test_small_outputs_remain_finite_and_overflowing_outputs_are_rejected(self):
        small_cbar = np.full((1, 1, 1), 1.0e-100 + 0.0j)
        small_dense = dense_exchange(small_cbar, np.ones((1, 1)))
        small_thc = thc_exchange(
            np.full((1, 1), 1.0e-100), np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
        )
        np.testing.assert_allclose(small_dense, small_thc, rtol=1.0e-15, atol=0.0)
        self.assertGreater(abs(small_dense[0, 0]), 0.0)

        with self.assertRaises(ValueError):
            dense_exchange(np.full((1, 1, 1), 1.0e200), np.ones((1, 1)))
        with self.assertRaises(ValueError):
            thc_exchange(
                np.full((1, 1), 1.0e200),
                np.ones((1, 1)),
                np.ones((1, 1)),
                np.ones((1, 1)),
            )

    def test_large_finite_nonhermitian_diagnostics_do_not_overflow(self):
        target = np.complex128(1.3e308 + 1.3e308j)
        metric = np.asarray([[-target]], dtype=np.complex128)

        dense = dense_exchange(
            np.ones((1, 1, 1), dtype=np.complex128), metric, diagnostics=True
        )
        thc = thc_exchange(
            np.ones((1, 1), dtype=np.complex128),
            np.ones((1, 1), dtype=np.complex128),
            np.ones((1, 1), dtype=np.complex128),
            metric,
            diagnostics=True,
        )

        for result in (dense, thc):
            self.assertTrue(np.isfinite(result.matrix).all())
            self.assertEqual(result.matrix[0, 0], target)
            self.assertTrue(np.isfinite(result.hermiticity_rel))
            self.assertGreater(result.hermiticity_rel, 0.0)


if __name__ == "__main__":
    unittest.main()
