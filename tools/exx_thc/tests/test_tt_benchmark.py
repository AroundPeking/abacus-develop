from __future__ import annotations

from dataclasses import replace
import itertools
import json
import unittest

import numpy as np

from exx_thc.supercell import occupied_exchange
from exx_thc.tt_benchmark import scan_exx_tt_routes, select_tt_point


class EXXTTBenchmarkTest(unittest.TestCase):
    @staticmethod
    def fixture():
        rng = np.random.default_rng(813)
        g1 = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
        g2 = rng.normal(size=(2, 4, 2)) + 1j * rng.normal(size=(2, 4, 2))
        g3 = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
        coefficient = np.einsum("ma,aib,bj->mij", g1, g2, g3)
        trial = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        metric = trial @ trial.conj().T + 0.5 * np.eye(5)
        occupied = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        return coefficient, metric, occupied

    def test_scans_three_routes_all_orders_and_tolerances(self):
        coefficient, metric, occupied = self.fixture()

        result = scan_exx_tt_routes(
            coefficient, metric, occupied, tolerances=(0.0, 1.0e-6), repeats=1
        )

        self.assertEqual(len(result.points), 3 * 6 * 2)
        self.assertEqual({point.route for point in result.points}, {"C", "B", "X"})
        self.assertEqual(
            {point.order for point in result.points},
            set(itertools.permutations((0, 1, 2))),
        )
        self.assertEqual({point.relative_tol for point in result.points}, {0.0, 1.0e-6})
        self.assertGreaterEqual(result.metric_setup_seconds, 0.0)
        self.assertIsNotNone(result.selected)
        json.dumps(result.to_json_dict(), allow_nan=False)

    def test_zero_tolerance_every_route_matches_complex_occupied_exchange(self):
        coefficient, metric, occupied = self.fixture()
        reference = occupied_exchange(coefficient, metric, occupied).matrix

        result = scan_exx_tt_routes(
            coefficient, metric, occupied, tolerances=(0.0,), repeats=1
        )

        for point in result.points:
            with self.subTest(route=point.route, order=point.order):
                np.testing.assert_allclose(
                    point.matrix, reference, rtol=2.0e-12, atol=2.0e-12
                )
                self.assertLess(point.h_rel_fro, 2.0e-12)
                self.assertLess(point.reconstruction_rel_fro, 2.0e-12)
                self.assertEqual(
                    point.dense_elements,
                    int(np.prod(point.canonical_shape)),
                )
                self.assertEqual(point.core_elements, sum(point.core_shapes_elements))
                self.assertAlmostEqual(
                    point.compression_ratio,
                    point.dense_elements / point.core_elements,
                    places=15,
                )
                self.assertGreaterEqual(point.setup_seconds, 0.0)
                self.assertGreaterEqual(point.exx_seconds, 0.0)
                if point.route == "C":
                    self.assertEqual(point.steady_seconds, point.exx_seconds)
                else:
                    self.assertEqual(
                        point.steady_seconds,
                        point.setup_seconds + point.exx_seconds,
                    )

    def test_selection_uses_accuracy_time_storage_and_static_route_tie_breaks(self):
        coefficient, metric, occupied = self.fixture()
        base = scan_exx_tt_routes(
            coefficient, metric, occupied, tolerances=(0.0,), repeats=1
        ).points[0]
        inaccurate = replace(
            base, route="X", h_rel_fro=2.0e-8, steady_seconds=0.1
        )
        fastest = replace(
            base,
            route="B",
            h_rel_fro=1.0e-9,
            steady_seconds=1.0,
            compression_ratio=5.0,
        )
        within_ten_percent = replace(
            base,
            route="X",
            h_rel_fro=1.0e-9,
            steady_seconds=1.09,
            compression_ratio=8.0,
        )
        outside_ten_percent = replace(
            base,
            route="C",
            h_rel_fro=1.0e-9,
            steady_seconds=1.11,
            compression_ratio=20.0,
        )

        selected = select_tt_point(
            (inaccurate, fastest, within_ten_percent, outside_ten_percent)
        )

        self.assertIs(selected, within_ten_percent)
        static = replace(within_ten_percent, route="C")
        self.assertIs(select_tt_point((within_ten_percent, static)), static)
        self.assertIsNone(select_tt_point((inaccurate,)))

    def test_rejects_invalid_inputs_before_scanning(self):
        coefficient, metric, occupied = self.fixture()
        cases = [
            (coefficient[:, :, :3], metric, occupied, (0.0,), 1),
            (coefficient, metric[:4, :4], occupied, (0.0,), 1),
            (coefficient, metric, occupied[:3], (0.0,), 1),
            (coefficient, metric, occupied, (), 1),
            (coefficient, metric, occupied, (-1.0,), 1),
            (coefficient, metric, occupied, (np.nan,), 1),
            (coefficient, metric, occupied, (0.0,), 0),
            (coefficient, metric, occupied, (0.0,), True),
        ]
        for C, V, O, tolerances, repeats in cases:
            with self.subTest(
                C_shape=C.shape,
                V_shape=V.shape,
                O_shape=O.shape,
                tolerances=tolerances,
                repeats=repeats,
            ), self.assertRaises(ValueError):
                scan_exx_tt_routes(C, V, O, tolerances=tolerances, repeats=repeats)


if __name__ == "__main__":
    unittest.main()
