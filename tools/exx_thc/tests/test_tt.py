from __future__ import annotations

import unittest

import numpy as np

from exx_thc.tt import tt_svd_3


class TTThreeTest(unittest.TestCase):
    @staticmethod
    def known_rank_tensor():
        tensor = np.zeros((2, 3, 3), dtype=np.complex128)
        tensor[0, 0, 0] = 10.0 + 2.0j
        tensor[0, 1, 1] = 1.0 - 0.3j
        tensor[0, 2, 2] = 0.5 + 0.1j
        tensor[1, 0, 1] = 0.2j
        return tensor

    def test_recovers_known_complex_tt_ranks_and_tensor(self):
        tensor = self.known_rank_tensor()

        result = tt_svd_3(tensor, relative_tol=1.0e-12)

        self.assertEqual(result.ranks, (2, 3))
        self.assertEqual(result.g1.dtype, np.complex128)
        self.assertEqual(result.g2.dtype, np.complex128)
        self.assertEqual(result.g3.dtype, np.complex128)
        np.testing.assert_allclose(result.reconstruct(), tensor, rtol=1.0e-11, atol=1.0e-11)

    def test_truncation_obeys_error_bound_and_minimal_tail_budget(self):
        tensor = self.known_rank_tensor()
        relative_tolerance = 0.5

        result = tt_svd_3(tensor, relative_tolerance)

        self.assertEqual(result.ranks, (1, 1))
        actual_error_squared = float(
            np.vdot(tensor - result.reconstruct(), tensor - result.reconstruct()).real
        )
        self.assertLessEqual(actual_error_squared, result.error_bound * (1.0 + 1.0e-12))
        norm_squared = float(np.vdot(tensor, tensor).real)
        budget = relative_tolerance**2 * norm_squared / 2.0
        for singular_values, rank, discarded_weight in zip(
            result.spectra, result.ranks, result.discarded_weights
        ):
            tail_squared = float(np.dot(singular_values[rank:], singular_values[rank:]))
            self.assertLessEqual(tail_squared, budget * (1.0 + 1.0e-13))
            if rank > 1:
                previous_tail_squared = float(
                    np.dot(singular_values[rank - 1 :], singular_values[rank - 1 :])
                )
                self.assertGreater(previous_tail_squared, budget)
            self.assertAlmostEqual(discarded_weight, tail_squared / norm_squared, places=14)
        self.assertAlmostEqual(
            result.error_bound,
            norm_squared * sum(result.discarded_weights),
            places=12,
        )
        self.assertEqual(result.spectra[0].size, min(tensor.shape[0], np.prod(tensor.shape[1:])))
        self.assertEqual(
            result.spectra[1].size,
            min(result.ranks[0] * tensor.shape[1], tensor.shape[2]),
        )

    def test_zero_tensor_has_empty_ranks_and_reconstructs_shape(self):
        tensor = np.zeros((3, 2, 4), dtype=np.complex128)

        result = tt_svd_3(tensor, 0.0)

        self.assertEqual(result.ranks, (0, 0))
        self.assertEqual(result.g1.shape, (3, 0))
        self.assertEqual(result.g2.shape, (0, 2, 0))
        self.assertEqual(result.g3.shape, (0, 4))
        self.assertEqual(result.reconstruct().shape, tensor.shape)
        np.testing.assert_array_equal(result.reconstruct(), tensor)
        self.assertEqual(result.error_bound, 0.0)
        self.assertEqual(result.discarded_weights, (0.0, 0.0))

    def test_finite_large_tolerance_uses_minimum_nonzero_ranks(self):
        result = tt_svd_3(self.known_rank_tensor(), 1.0e308)

        self.assertEqual(result.ranks, (1, 1))

    def test_subnormal_squared_norm_scalar_remains_nonzero_rank(self):
        tensor = np.asarray([[[1.0e-200 + 0.0j]]], dtype=np.complex128)

        result = tt_svd_3(tensor, 0.0)

        self.assertEqual(result.ranks, (1, 1))
        np.testing.assert_allclose(result.reconstruct(), tensor, rtol=1.0e-15, atol=0.0)

    def test_zero_tolerance_preserves_small_independent_components(self):
        tensor = np.zeros((2, 1, 2), dtype=np.complex128)
        tensor[0, 0, 0] = 1.0e-160
        tensor[1, 0, 1] = 1.0e-162j

        result = tt_svd_3(tensor, 0.0)

        self.assertEqual(result.ranks, (2, 2))
        np.testing.assert_allclose(result.reconstruct(), tensor, rtol=1.0e-14, atol=0.0)
        self.assertEqual(result.error_bound, 0.0)

    def test_ranks_are_invariant_under_representable_global_scaling(self):
        tensor = self.known_rank_tensor()
        scaled = tensor * 1.0e-150

        for tolerance in (1.0e-12, 0.5):
            with self.subTest(tolerance=tolerance):
                self.assertEqual(
                    tt_svd_3(scaled, tolerance).ranks,
                    tt_svd_3(tensor, tolerance).ranks,
                )

    def test_underflowed_absolute_bound_is_positive_when_tail_is_discarded(self):
        tensor = np.zeros((2, 1, 2), dtype=np.complex128)
        tensor[0, 0, 0] = 1.0e-200
        tensor[1, 0, 1] = 5.0e-201j

        result = tt_svd_3(tensor, 1.0)

        self.assertEqual(result.ranks, (1, 1))
        self.assertAlmostEqual(result.discarded_weights[0], 0.2, places=15)
        self.assertEqual(result.discarded_weights[1], 0.0)
        self.assertGreater(result.error_bound, 0.0)
        self.assertTrue(np.isfinite(result.error_bound))

    def test_large_finite_complex64_is_promoted_and_reconstructed(self):
        tensor = np.full((1, 1, 2), 3.0e38 + 0.0j, dtype=np.complex64)

        result = tt_svd_3(tensor, 0.0)

        self.assertEqual(result.ranks, (1, 1))
        for array in (*result.spectra, result.g1, result.g2, result.g3):
            self.assertTrue(np.isfinite(array).all())
        self.assertEqual(result.g1.dtype, np.complex128)
        self.assertEqual(result.g2.dtype, np.complex128)
        self.assertEqual(result.g3.dtype, np.complex128)
        np.testing.assert_allclose(result.reconstruct(), tensor, rtol=2.0e-6, atol=0.0)

    def test_rejects_bad_shape_empty_nonfinite_tolerance_and_overflow_norm(self):
        valid = np.ones((2, 2, 2), dtype=np.complex128)
        cases = [
            (np.ones((2, 2), dtype=np.complex128), 0.0),
            (np.empty((2, 0, 3), dtype=np.complex128), 0.0),
            (np.full((2, 2, 2), np.nan + 0.0j), 0.0),
            (np.full((2, 2, 2), np.inf + 0.0j), 0.0),
            (np.full((2, 2, 2), 1.0e308 + 0.0j), 0.0),
            (valid, -1.0),
            (valid, np.nan),
            (valid, np.inf),
        ]
        for tensor, tolerance in cases:
            with self.subTest(shape=tensor.shape, tolerance=tolerance), self.assertRaises(
                ValueError
            ):
                tt_svd_3(tensor, tolerance)


if __name__ == "__main__":
    unittest.main()
