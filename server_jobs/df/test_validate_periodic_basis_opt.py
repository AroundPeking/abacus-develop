import unittest

import validate_periodic_basis_opt as validator


class WhiteningProbeLimitTest(unittest.TestCase):
    def test_uses_elementwise_infinity_norm_bound(self):
        retained_rank = 235
        declared_max_element_error = 4.489126877269119e-9
        sampled_error = 1.5238731160485709e-7

        limit = validator.whitening_probe_limit(
            retained_rank, declared_max_element_error
        )

        self.assertAlmostEqual(
            limit,
            retained_rank * declared_max_element_error + 1.0e-12,
        )
        self.assertLess(sampled_error, limit)

    def test_keeps_absolute_roundoff_floor(self):
        self.assertEqual(validator.whitening_probe_limit(1, 2.0e-9), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
