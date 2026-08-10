from __future__ import annotations

import unittest

import numpy as np

from exx_thc.thc import cp_als


def _exact_case():
    rng = np.random.default_rng(11)
    t = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    x = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    y = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    cbar = np.einsum("mr,ir,vr->miv", t.conj(), x, y)
    return np.asarray(cbar, dtype=np.complex128, order="C")


class AlgebraicTHCTest(unittest.TestCase):
    def test_known_complex_rank_three_has_small_best_seed_residual(self):
        cbar = _exact_case()
        results = [
            cp_als(cbar, 3, max_iter=500, rel_tol=1.0e-10, ridge=1.0e-14, seed=seed)
            for seed in (0, 1, 2)
        ]

        residuals = [result.residual for result in results]
        self.assertLess(min(residuals), 1.0e-8)
        self.assertLessEqual(min(residuals), max(residuals))
        best = results[int(np.argmin(residuals))]
        rebuilt = np.einsum("mr,ir,vr->miv", best.T.conj(), best.X, best.Y)
        np.testing.assert_allclose(rebuilt, cbar, rtol=1.0e-8, atol=1.0e-8)
        for factor in (best.T, best.X, best.Y):
            self.assertEqual(factor.dtype, np.complex128)
            self.assertTrue(factor.flags.c_contiguous)
            self.assertTrue(np.isfinite(factor).all())
        for column in range(3):
            a_column = best.T[:, column].conj()
            pivot = a_column[int(np.argmax(np.abs(a_column)))]
            self.assertGreaterEqual(pivot.real, 0.0)
            self.assertLess(abs(pivot.imag), 1.0e-12)

    def test_same_seed_is_strictly_reproducible(self):
        cbar = _exact_case()
        first = cp_als(cbar, 3, max_iter=20, rel_tol=1.0e-8, seed=7)
        second = cp_als(cbar, 3, max_iter=20, rel_tol=1.0e-8, seed=7)

        np.testing.assert_array_equal(first.T, second.T)
        np.testing.assert_array_equal(first.X, second.X)
        np.testing.assert_array_equal(first.Y, second.Y)
        self.assertEqual(first.residual, second.residual)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.converged, second.converged)

    def test_seed_changes_full_svd_initial_subspace_orientation(self):
        cbar = _exact_case()
        results = [
            cp_als(cbar, 3, max_iter=1, rel_tol=0.0, ridge=1.0e-14, seed=seed)
            for seed in (0, 1, 2)
        ]
        repeated = cp_als(cbar, 3, max_iter=1, rel_tol=0.0, ridge=1.0e-14, seed=0)

        np.testing.assert_array_equal(results[0].T, repeated.T)
        np.testing.assert_array_equal(results[0].X, repeated.X)
        np.testing.assert_array_equal(results[0].Y, repeated.Y)
        self.assertTrue(
            any(
                not np.array_equal(results[0].T, candidate.T)
                or not np.array_equal(results[0].X, candidate.X)
                or not np.array_equal(results[0].Y, candidate.Y)
                for candidate in results[1:]
            )
        )

    def test_converged_requires_two_consecutive_residual_changes(self):
        tensor = np.asarray([[[1.0 + 0.4j, -0.3j]]], dtype=np.complex128)

        two = cp_als(tensor, 1, max_iter=2, rel_tol=2.0, ridge=0.0, seed=0)
        three = cp_als(tensor, 1, max_iter=3, rel_tol=2.0, ridge=0.0, seed=0)

        self.assertFalse(two.converged)
        self.assertEqual(two.iterations, 2)
        self.assertTrue(three.converged)
        self.assertEqual(three.iterations, 3)

    def test_zero_tensor_returns_explicit_zero_factors(self):
        tensor = np.zeros((3, 2, 4), dtype=np.complex128)

        result = cp_als(tensor, 2, seed=1)

        self.assertEqual(result.T.shape, (3, 2))
        self.assertEqual(result.X.shape, (2, 2))
        self.assertEqual(result.Y.shape, (4, 2))
        self.assertEqual(result.residual, 0.0)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(result.T, np.zeros_like(result.T))
        np.testing.assert_array_equal(result.X, np.zeros_like(result.X))
        np.testing.assert_array_equal(result.Y, np.zeros_like(result.Y))

    def test_global_scaling_avoids_norm_underflow_and_overflow(self):
        base = np.asarray([[[1.0 + 0.2j, -0.4 + 0.3j]]], dtype=np.complex128)
        for scale in (1.0e-200, 1.0e200):
            with self.subTest(scale=scale):
                tensor = np.asarray(base * scale, dtype=np.complex128, order="C")
                result = cp_als(tensor, 1, max_iter=20, rel_tol=1.0e-12, ridge=0.0, seed=0)
                rebuilt = np.einsum("mr,ir,vr->miv", result.T.conj(), result.X, result.Y)
                self.assertTrue(np.isfinite(rebuilt).all())
                np.testing.assert_allclose(rebuilt, tensor, rtol=1.0e-11, atol=0.0)

    def test_promotes_real_complex64_fortran_and_noncontiguous_inputs(self):
        first = np.asarray([1.0 + 0.2j, -0.4j])
        second = np.asarray([0.7 - 0.1j, -0.3 + 0.5j])
        third = np.asarray([1.2j, 0.6 + 0.2j])
        base = np.einsum("m,i,v->miv", first, second, third)
        real_base = np.einsum("m,i,v->miv", [1.0, 0.4], [0.7, -0.2], [0.3, 0.9])
        variants = [
            real_base,
            base.astype(np.complex64),
            np.asfortranarray(base),
            base[:, :, ::-1],
        ]
        for tensor in variants:
            with self.subTest(dtype=tensor.dtype, c_contiguous=tensor.flags.c_contiguous):
                result = cp_als(tensor, 1, max_iter=20, rel_tol=1.0e-10, ridge=0.0, seed=0)
                rebuilt = np.einsum("mr,ir,vr->miv", result.T.conj(), result.X, result.Y)
                np.testing.assert_allclose(rebuilt, tensor, rtol=2.0e-6, atol=1.0e-12)
                for factor in (result.T, result.X, result.Y):
                    self.assertEqual(factor.dtype, np.complex128)
                    self.assertTrue(factor.flags.c_contiguous)

    def test_rejects_invalid_tensor_and_parameters(self):
        valid = np.ones((2, 2, 2), dtype=np.complex128, order="C")
        cases = [
            (np.ones((2, 2), dtype=np.complex128), {}),
            (np.empty((2, 0, 2), dtype=np.complex128), {}),
            (np.full((2, 2, 2), np.nan + 0.0j), {}),
            (np.full((2, 2, 2), np.inf + 0.0j), {}),
            (np.full((2, 2, 2), "not numeric", dtype=object), {}),
            (valid, {"rank": 0}),
            (valid, {"rank": 1.5}),
            (valid, {"max_iter": 0}),
            (valid, {"max_iter": 1.5}),
            (valid, {"rel_tol": -1.0}),
            (valid, {"rel_tol": np.inf}),
            (valid, {"ridge": -1.0}),
            (valid, {"ridge": np.nan}),
            (valid, {"seed": -1}),
            (valid, {"seed": 1.5}),
        ]
        for tensor, overrides in cases:
            arguments = dict(rank=1, max_iter=5, rel_tol=1.0e-8, ridge=1.0e-12, seed=0)
            arguments.update(overrides)
            with self.subTest(overrides=overrides, shape=tensor.shape), self.assertRaises(
                ValueError
            ):
                cp_als(tensor, **arguments)


if __name__ == "__main__":
    unittest.main()
