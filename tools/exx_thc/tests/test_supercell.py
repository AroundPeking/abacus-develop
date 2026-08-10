from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import numpy as np

from exx_thc.io import BlockKey
from exx_thc.supercell import (
    assemble_pair_coefficient,
    assemble_translation_matrix,
    infer_supercell_layout,
)


class SupercellAssemblyTest(unittest.TestCase):
    def one_atom_blocks(self):
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[[2.0]]], dtype=np.float64),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[[3.0]]], dtype=np.float64),
        }
        matrix_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[5.0]], dtype=np.float64),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[7.0]], dtype=np.float64),
        }
        return coefficient_blocks, matrix_blocks

    def test_c_expansion_contains_both_ao_orders_and_all_cells(self) -> None:
        period = (2, 1, 1)
        coefficient_blocks, matrix_blocks = self.one_atom_blocks()

        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, period
        )
        coefficient = assemble_pair_coefficient(
            coefficient_blocks, layout, max_elements=64
        )

        expected = np.zeros((2, 2, 2), dtype=np.float64)
        for cell in range(2):
            expected[cell, cell, cell] += 4.0
            other = (cell - 1) % 2
            expected[cell, cell, other] += 3.0
            expected[cell, other, cell] += 3.0
        np.testing.assert_array_equal(coefficient, expected)

    def test_v_and_d_expansion_follow_outer_cell_plus_r(self) -> None:
        period = (2, 1, 1)
        coefficient_blocks, matrix_blocks = self.one_atom_blocks()
        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, period
        )

        for space in ("ao", "auxiliary"):
            with self.subTest(space=space):
                matrix = assemble_translation_matrix(
                    matrix_blocks, layout, space=space, max_elements=16
                )
                np.testing.assert_array_equal(
                    matrix, np.asarray([[5.0, 7.0], [7.0, 5.0]])
                )

    def test_translation_aliases_are_summed(self) -> None:
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[[1.0]]]),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[[3.0]]]),
            BlockKey(0, 0, (1, 0, 0)): np.asarray([[[4.0]]]),
        }
        matrix_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[2.0]]),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[5.0]]),
            BlockKey(0, 0, (1, 0, 0)): np.asarray([[7.0]]),
        }
        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, (2, 1, 1)
        )

        coefficient = assemble_pair_coefficient(
            coefficient_blocks, layout, max_elements=8
        )
        expected_coefficient = np.zeros((2, 2, 2))
        expected_coefficient[0, 0, 0] = 2.0
        expected_coefficient[1, 1, 1] = 2.0
        expected_coefficient[0, 0, 1] = 7.0
        expected_coefficient[0, 1, 0] = 7.0
        expected_coefficient[1, 1, 0] = 7.0
        expected_coefficient[1, 0, 1] = 7.0
        np.testing.assert_array_equal(coefficient, expected_coefficient)

        matrix = assemble_translation_matrix(
            matrix_blocks, layout, space="ao", max_elements=4
        )
        np.testing.assert_array_equal(matrix, np.asarray([[2.0, 12.0], [12.0, 2.0]]))

    def test_multi_atom_layout_has_cell_major_immutable_offsets(self) -> None:
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((1, 2, 2)),
            BlockKey(1, 1, (0, 0, 0)): np.ones((2, 1, 1)),
        }
        metric_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1)),
            BlockKey(1, 1, (0, 0, 0)): np.ones((2, 2)),
        }
        density_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((2, 2)),
            BlockKey(1, 1, (0, 0, 0)): np.ones((1, 1)),
        }

        layout = infer_supercell_layout(
            coefficient_blocks, metric_blocks, density_blocks, (2, 1, 1)
        )

        self.assertEqual(layout.period, (2, 1, 1))
        self.assertEqual(layout.cells, ((0, 0, 0), (1, 0, 0)))
        self.assertEqual(layout.atoms, (0, 1))
        self.assertEqual(dict(layout.ao_dimensions), {0: 2, 1: 1})
        self.assertEqual(dict(layout.auxiliary_dimensions), {0: 1, 1: 2})
        self.assertEqual(
            dict(layout.ao_offsets),
            {
                ((0, 0, 0), 0): 0,
                ((0, 0, 0), 1): 2,
                ((1, 0, 0), 0): 3,
                ((1, 0, 0), 1): 5,
            },
        )
        self.assertEqual(
            dict(layout.auxiliary_offsets),
            {
                ((0, 0, 0), 0): 0,
                ((0, 0, 0), 1): 1,
                ((1, 0, 0), 0): 3,
                ((1, 0, 0), 1): 4,
            },
        )
        self.assertEqual((layout.nao_supercell, layout.naux_supercell), (6, 6))
        with self.assertRaises(FrozenInstanceError):
            layout.nao_supercell = 9
        with self.assertRaises(TypeError):
            layout.ao_dimensions[0] = 9

    def test_complex_blocks_produce_complex_dense_arrays(self) -> None:
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[[1.0 + 2.0j]]])
        }
        metric_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[2.0]], dtype=np.float64)
        }
        density_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[3.0 + 4.0j]])
        }
        layout = infer_supercell_layout(
            coefficient_blocks, metric_blocks, density_blocks, (1, 1, 1)
        )

        coefficient = assemble_pair_coefficient(
            coefficient_blocks, layout, max_elements=1
        )
        density = assemble_translation_matrix(
            density_blocks, layout, space="ao", max_elements=1
        )

        self.assertEqual(coefficient.dtype, np.dtype(np.complex128))
        self.assertEqual(density.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(coefficient, np.asarray([[[2.0 + 4.0j]]]))
        np.testing.assert_array_equal(density, np.asarray([[3.0 + 4.0j]]))

    def test_rejects_nonfinite_onsite_coefficient_accumulation(self) -> None:
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray(
                [[[1.0e308]]], dtype=np.float64
            )
        }
        matrix_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[1.0]], dtype=np.float64)
        }
        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, (1, 1, 1)
        )

        with self.assertRaisesRegex(ValueError, "finite|overflow"):
            assemble_pair_coefficient(
                coefficient_blocks, layout, max_elements=1
            )

    def test_rejects_nonfinite_translation_alias_accumulation(self) -> None:
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[[1.0]]], dtype=np.float64)
        }
        translation_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray(
                [[1.0e308]], dtype=np.float64
            ),
            BlockKey(0, 0, (1, 0, 0)): np.asarray(
                [[1.0e308]], dtype=np.float64
            ),
        }
        layout = infer_supercell_layout(
            coefficient_blocks,
            translation_blocks,
            translation_blocks,
            (1, 1, 1),
        )

        with self.assertRaisesRegex(ValueError, "finite|overflow"):
            assemble_translation_matrix(
                translation_blocks, layout, space="ao", max_elements=1
            )

    def test_dense_memory_guard_and_max_elements_validation(self) -> None:
        coefficient_blocks, matrix_blocks = self.one_atom_blocks()
        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, (2, 1, 1)
        )

        with self.assertRaisesRegex(
            ValueError, "^dense supercell allocation exceeds max_elements$"
        ):
            assemble_pair_coefficient(coefficient_blocks, layout, max_elements=7)
        with self.assertRaisesRegex(
            ValueError, "^dense supercell allocation exceeds max_elements$"
        ):
            assemble_translation_matrix(
                matrix_blocks, layout, space="ao", max_elements=3
            )
        for invalid in (0, -1, 1.5, True):
            with self.subTest(max_elements=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                assemble_pair_coefficient(
                    coefficient_blocks, layout, max_elements=invalid
                )

    def test_rejects_invalid_period_atom_ids_and_missing_dimensions(self) -> None:
        coefficient_blocks, matrix_blocks = self.one_atom_blocks()
        for period in ((2, 1), (2, 0, 1), (2, -1, 1), (2, 1.5, 1), [2, 1, 1]):
            with self.subTest(period=period), self.assertRaises(
                (TypeError, ValueError)
            ):
                infer_supercell_layout(
                    coefficient_blocks, matrix_blocks, matrix_blocks, period
                )

        invalid_atom_maps = (
            (
                {BlockKey(-1, -1, (0, 0, 0)): np.ones((1, 1, 1))},
                {BlockKey(-1, -1, (0, 0, 0)): np.ones((1, 1))},
            ),
            (
                {
                    BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1, 1)),
                    BlockKey(2, 2, (0, 0, 0)): np.ones((1, 1, 1)),
                },
                {
                    BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1)),
                    BlockKey(2, 2, (0, 0, 0)): np.ones((1, 1)),
                },
            ),
        )
        for coefficient, matrix in invalid_atom_maps:
            with self.assertRaises(ValueError):
                infer_supercell_layout(coefficient, matrix, matrix, (1, 1, 1))

        with self.assertRaises(ValueError):
            infer_supercell_layout({}, {}, {}, (1, 1, 1))

        missing_c_dimensions = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1, 1))
        }
        matrix_with_atom_one = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1)),
            BlockKey(1, 1, (0, 0, 0)): np.ones((1, 1)),
        }
        with self.assertRaisesRegex(ValueError, "missing .* dimension"):
            infer_supercell_layout(
                missing_c_dimensions,
                matrix_with_atom_one,
                matrix_with_atom_one,
                (1, 1, 1),
            )

    def test_large_noncontiguous_atom_id_is_rejected_without_large_range(self) -> None:
        atom = (1 << 31) - 1
        coefficient_blocks = {
            BlockKey(atom, atom, (0, 0, 0)): np.ones((1, 1, 1))
        }
        matrix_blocks = {
            BlockKey(atom, atom, (0, 0, 0)): np.ones((1, 1))
        }

        with mock.patch(
            "exx_thc.supercell.range",
            side_effect=AssertionError("atom maximum was used to construct a range"),
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "atom ids"):
                infer_supercell_layout(
                    coefficient_blocks,
                    matrix_blocks,
                    matrix_blocks,
                    (1, 1, 1),
                )

    def test_rejects_dtype_rank_nonfinite_shape_and_key_errors(self) -> None:
        coefficient_blocks, matrix_blocks = self.one_atom_blocks()
        invalid_coefficients = {
            "dtype": {BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1, 1), dtype=np.float32)},
            "rank": {BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1))},
            "nonfinite": {
                BlockKey(0, 0, (0, 0, 0)): np.asarray([[[np.nan]]])
            },
            "array": {BlockKey(0, 0, (0, 0, 0)): [[[1.0]]]},
            "key": {(0, 0, (0, 0, 0)): np.ones((1, 1, 1))},
            "R rank": {BlockKey(0, 0, (0, 0)): np.ones((1, 1, 1))},
            "R dtype": {BlockKey(0, 0, (0, 0.5, 0)): np.ones((1, 1, 1))},
        }
        for name, blocks in invalid_coefficients.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                infer_supercell_layout(blocks, matrix_blocks, matrix_blocks, (2, 1, 1))

        for name, invalid_matrix in {
            "dtype": np.ones((1, 1), dtype=np.int64),
            "rank": np.ones((1, 1, 1), dtype=np.float64),
            "nonfinite": np.asarray([[np.inf]], dtype=np.float64),
        }.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                infer_supercell_layout(
                    coefficient_blocks,
                    {BlockKey(0, 0, (0, 0, 0)): invalid_matrix},
                    matrix_blocks,
                    (2, 1, 1),
                )

        inconsistent_c = dict(coefficient_blocks)
        inconsistent_c[BlockKey(0, 0, (1, 0, 0))] = np.ones((2, 1, 1))
        with self.assertRaisesRegex(ValueError, "inconsistent auxiliary dimension"):
            infer_supercell_layout(
                inconsistent_c, matrix_blocks, matrix_blocks, (2, 1, 1)
            )

        bad_metric_shape = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((2, 1), dtype=np.float64)
        }
        with self.assertRaisesRegex(ValueError, "metric block shape"):
            infer_supercell_layout(
                coefficient_blocks,
                bad_metric_shape,
                matrix_blocks,
                (2, 1, 1),
            )

        bad_density_shape = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((2, 1), dtype=np.float64)
        }
        with self.assertRaisesRegex(ValueError, "density block shape"):
            infer_supercell_layout(
                coefficient_blocks,
                matrix_blocks,
                bad_density_shape,
                (2, 1, 1),
            )

    def test_assembly_rejects_invalid_blocks_and_space_before_allocation(self) -> None:
        coefficient_blocks, matrix_blocks = self.one_atom_blocks()
        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, (2, 1, 1)
        )

        with mock.patch(
            "exx_thc.supercell.np.zeros",
            side_effect=AssertionError("dense allocation was attempted"),
        ):
            with self.assertRaisesRegex(ValueError, "coefficient block shape"):
                assemble_pair_coefficient(
                    {BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1, 2))},
                    layout,
                    max_elements=64,
                )
            with self.assertRaisesRegex(ValueError, "unknown atom"):
                assemble_translation_matrix(
                    {BlockKey(1, 1, (0, 0, 0)): np.ones((1, 1))},
                    layout,
                    space="ao",
                    max_elements=16,
                )
            with self.assertRaisesRegex(ValueError, "space"):
                assemble_translation_matrix(
                    matrix_blocks,
                    layout,
                    space="orbital",
                    max_elements=16,
                )


if __name__ == "__main__":
    unittest.main()
