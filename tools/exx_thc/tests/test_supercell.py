from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import numpy as np

import exx_thc.supercell as supercell
from exx_thc.io import BlockKey
from exx_thc.supercell import (
    ExchangeResult,
    assemble_pair_coefficient,
    assemble_translation_matrix,
    direct_exchange,
    infer_supercell_layout,
    occupied_exchange,
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


class BlockExtractionTest(unittest.TestCase):
    def _layout(self, period=(1, 1, 1), dimensions=(1,)):
        coefficient_blocks = {
            BlockKey(atom, atom, (0, 0, 0)): np.ones(
                (1, dimension, dimension), dtype=np.float64
            )
            for atom, dimension in enumerate(dimensions)
        }
        metric_blocks = {
            BlockKey(atom, atom, (0, 0, 0)): np.ones(
                (1, 1), dtype=np.float64
            )
            for atom, _dimension in enumerate(dimensions)
        }
        density_blocks = {
            BlockKey(atom, atom, (0, 0, 0)): np.eye(
                dimension, dtype=np.float64
            )
            for atom, dimension in enumerate(dimensions)
        }
        return infer_supercell_layout(
            coefficient_blocks, metric_blocks, density_blocks, period
        )

    def _extract(self):
        self.assertIn("extract_reference_cell_blocks", supercell.__all__)
        function = getattr(supercell, "extract_reference_cell_blocks", None)
        self.assertIsNotNone(function)
        return function

    def _dotc(self):
        self.assertIn("map_dotc", supercell.__all__)
        function = getattr(supercell, "map_dotc", None)
        self.assertIsNotNone(function)
        return function

    @staticmethod
    def _canonical_cell(cell, period):
        return tuple(
            residue if residue <= (extent - 1) // 2 else residue - extent
            for residue, extent in zip(cell, period)
        )

    def test_extracts_all_ordered_pair_translation_blocks_and_roundtrips(self):
        extract = self._extract()
        layout = self._layout(period=(3, 2, 1), dimensions=(2, 1))
        expected_blocks = {}
        expected_keys = []
        for ia1 in layout.atoms:
            for ia2 in layout.atoms:
                shape = (
                    layout.ao_dimensions[ia1],
                    layout.ao_dimensions[ia2],
                )
                for cell_index, residue in enumerate(layout.cells):
                    cell = self._canonical_cell(residue, layout.period)
                    key = BlockKey(ia1, ia2, cell)
                    expected_keys.append(key)
                    base = 100 * ia1 + 20 * ia2 + cell_index + 1
                    values = np.arange(np.prod(shape), dtype=np.float64).reshape(
                        shape
                    )
                    block = values + base + 1j * (0.25 * values - base)
                    if ia1 == 1 and ia2 == 0 and cell == (-1, -1, 0):
                        block = np.zeros(shape, dtype=np.complex128)
                    expected_blocks[key] = np.asarray(
                        block, dtype=np.complex128, order="C"
                    )
        dense = assemble_translation_matrix(
            expected_blocks,
            layout,
            space="ao",
            max_elements=layout.nao_supercell**2,
        )
        dense_before = dense.copy()

        extracted = extract(dense, layout, "complex128")

        self.assertEqual(list(extracted), expected_keys)
        self.assertIn(BlockKey(0, 0, (0, -1, 0)), extracted)
        self.assertNotIn(BlockKey(0, 0, (0, 1, 0)), extracted)
        for key, expected in expected_blocks.items():
            with self.subTest(key=key):
                np.testing.assert_array_equal(extracted[key], expected)
        reassembled = assemble_translation_matrix(
            extracted,
            layout,
            space="ao",
            max_elements=layout.nao_supercell**2,
        )
        np.testing.assert_array_equal(reassembled, dense)
        np.testing.assert_array_equal(dense, dense_before)

    def test_complex_output_preserves_phase_dtype_contiguity_and_copies(self):
        extract = self._extract()
        layout = self._layout(dimensions=(2,))
        matrix = np.asfortranarray(
            np.asarray(
                [[1.0 + 2.0j, 3.0 - 4.0j], [-5.0 + 6.0j, 7.0 + 0.5j]],
                dtype=np.complex128,
            )
        )
        before = matrix.copy()

        blocks = extract(matrix, layout, "complex128")
        block = blocks[BlockKey(0, 0, (0, 0, 0))]

        np.testing.assert_array_equal(block, matrix)
        self.assertEqual(block.dtype, np.dtype(np.complex128))
        self.assertTrue(block.flags.c_contiguous)
        self.assertFalse(np.shares_memory(block, matrix))
        np.testing.assert_array_equal(matrix, before)

    def test_real_output_accepts_only_globally_negligible_imaginary_part(self):
        extract = self._extract()
        layout = self._layout(dimensions=(2,))
        within = np.asarray(
            [[2.0 + 1.9e-12j, -1.0], [0.5, -2.0 - 1.9e-12j]],
            dtype=np.complex128,
        )
        before = within.copy()

        block = extract(within, layout, "real64")[
            BlockKey(0, 0, (0, 0, 0))
        ]

        np.testing.assert_array_equal(block, within.real)
        self.assertEqual(block.dtype, np.dtype(np.float64))
        self.assertTrue(block.flags.c_contiguous)
        self.assertFalse(np.shares_memory(block, within))
        np.testing.assert_array_equal(within, before)

        outside = within.copy()
        outside[0, 0] = 2.0 + 2.1e-12j
        with self.assertRaisesRegex(ValueError, "imaginary|real64"):
            extract(outside, layout, "real64")

    def test_extract_rejects_invalid_scalar_layout_shape_and_nonfinite_matrix(self):
        extract = self._extract()
        layout = self._layout(dimensions=(2,))
        matrix = np.eye(2, dtype=np.complex128)

        invalid_scalars = (
            "float64",
            "complex64",
            "",
            None,
            np.asarray(["real64"]),
            np.asarray(["complex128"]),
            b"real64",
            ["real64"],
        )
        for scalar in invalid_scalars:
            with self.subTest(scalar=scalar), self.assertRaises(
                (TypeError, ValueError)
            ):
                extract(matrix, layout, scalar)
        for invalid_layout in (None, object(), (1, 1, 1)):
            with self.subTest(layout=invalid_layout), self.assertRaises(TypeError):
                extract(matrix, invalid_layout, "complex128")
        for shape in ((2,), (2, 2, 1), (3, 3), (1, 2)):
            with self.subTest(shape=shape), self.assertRaisesRegex(
                ValueError, "shape"
            ):
                extract(np.ones(shape), layout, "complex128")
        for value in (np.nan, np.inf, -np.inf):
            nonfinite = matrix.copy()
            nonfinite[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                extract(nonfinite, layout, "complex128")

    def test_map_dotc_conjugates_density_and_ignores_unmatched_keys(self):
        dotc = self._dotc()
        matching = BlockKey(0, 1, (0, -1, 0))
        density_only = BlockKey(1, 0, (0, 0, 0))
        h_only = BlockKey(1, 1, (0, 0, 0))
        density = {
            matching: np.asarray(
                [[1.0 + 2.0j, -3.0 + 0.5j]], dtype=np.complex128
            ),
            density_only: np.asarray([[1.0e200]], dtype=np.float64),
        }
        h_blocks = {
            matching: np.asarray(
                [[2.0 - 1.0j, 0.25 + 4.0j]], dtype=np.complex128
            ),
            h_only: np.asarray([[1.0e200]], dtype=np.float64),
        }
        density_before = {key: value.copy() for key, value in density.items()}
        h_before = {key: value.copy() for key, value in h_blocks.items()}
        expected = sum(
            left.conjugate() * right
            for left, right in zip(
                density[matching].flat, h_blocks[matching].flat
            )
        )

        result = dotc(density, h_blocks)

        self.assertIs(type(result), complex)
        self.assertAlmostEqual(result, expected)
        for original, snapshot in (
            (density, density_before),
            (h_blocks, h_before),
        ):
            self.assertEqual(set(original), set(snapshot))
            for key in original:
                np.testing.assert_array_equal(original[key], snapshot[key])

    def test_map_dotc_uses_sorted_key_order(self):
        dotc = self._dotc()
        keys = [BlockKey(index, index, (0, 0, 0)) for index in range(3)]
        density = {
            keys[0]: np.ones((1, 1), dtype=np.float64),
            keys[2]: np.ones((1, 1), dtype=np.float64),
            keys[1]: np.ones((1, 1), dtype=np.float64),
        }
        h_blocks = {
            keys[0]: np.asarray([[1.0e16]], dtype=np.float64),
            keys[2]: np.asarray([[1.0]], dtype=np.float64),
            keys[1]: np.asarray([[-1.0e16]], dtype=np.float64),
        }

        self.assertEqual(dotc(density, h_blocks), 1.0 + 0.0j)

    def test_map_dotc_empty_intersection_returns_python_complex_zero(self):
        dotc = self._dotc()
        density = {
            BlockKey(0, 0, (0, 0, 0)): np.ones((1, 1), dtype=np.float64)
        }
        h_blocks = {
            BlockKey(0, 0, (1, 0, 0)): np.ones((1, 1), dtype=np.float64)
        }

        result = dotc(density, h_blocks)

        self.assertIs(type(result), complex)
        self.assertEqual(result, 0j)
        self.assertEqual(dotc({}, {}), 0j)

    def test_map_dotc_rejects_shape_mapping_key_value_rank_dtype_and_nonfinite(self):
        dotc = self._dotc()
        key = BlockKey(0, 0, (0, 0, 0))
        valid = {key: np.ones((1, 1), dtype=np.float64)}
        with self.assertRaisesRegex(ValueError, "shape"):
            dotc(valid, {key: np.ones((1, 2), dtype=np.float64)})

        for bad_mapping in (None, [], ((key, np.ones((1, 1))),)):
            with self.subTest(mapping=bad_mapping):
                with self.assertRaises(TypeError):
                    dotc(bad_mapping, valid)
                with self.assertRaises(TypeError):
                    dotc(valid, bad_mapping)

        bad_maps = {
            "key": {"not-a-block-key": np.ones((1, 1), dtype=np.float64)},
            "value": {key: [[1.0]]},
            "rank": {key: np.ones((1,), dtype=np.float64)},
            "dtype": {key: np.ones((1, 1), dtype=np.float32)},
            "nonfinite": {key: np.asarray([[np.nan]], dtype=np.float64)},
        }
        for name, bad_map in bad_maps.items():
            with self.subTest(name=name, side="density"):
                with self.assertRaises((TypeError, ValueError)):
                    dotc(bad_map, valid)
            with self.subTest(name=name, side="H"):
                with self.assertRaises((TypeError, ValueError)):
                    dotc(valid, bad_map)

    def test_map_dotc_rejects_malformed_block_key_fields_before_sorting(self):
        dotc = self._dotc()
        block = np.ones((1, 1), dtype=np.float64)
        malformed = (
            (BlockKey(-1, 0, (0, 0, 0)), ValueError, "ia1"),
            (BlockKey("atom", 0, (0, 0, 0)), TypeError, "ia1"),
            (BlockKey(True, 0, (0, 0, 0)), TypeError, "ia1"),
            (BlockKey(0, 0, (0, 0)), ValueError, "R"),
            (BlockKey(0, 0, (0, "R", 0)), TypeError, "R"),
            (BlockKey(0, 0, (0, True, 0)), TypeError, "R"),
        )
        for key, error_type, message in malformed:
            blocks = {key: block}
            with self.subTest(key=key), self.assertRaisesRegex(
                error_type, message
            ):
                dotc(blocks, blocks)

        string_atom = BlockKey("atom", 0, (0, 0, 0))
        string_cell = BlockKey(0, 0, (0, "R", 0))
        heterogeneous_orders = (
            ((string_atom, string_cell), "ia1"),
            ((string_cell, string_atom), "R"),
        )
        for keys, message in heterogeneous_orders:
            blocks = {key: block for key in keys}
            with self.subTest(keys=keys), self.assertRaisesRegex(
                TypeError, message
            ):
                dotc(blocks, dict(blocks))

    def test_map_dotc_rejects_finite_product_and_sum_overflow(self):
        dotc = self._dotc()
        key0 = BlockKey(0, 0, (0, 0, 0))
        key1 = BlockKey(1, 1, (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "finite|overflow"):
            dotc(
                {key0: np.asarray([[1.0e308]], dtype=np.float64)},
                {key0: np.asarray([[1.0e308]], dtype=np.float64)},
            )
        density = {
            key0: np.ones((1, 1), dtype=np.float64),
            key1: np.ones((1, 1), dtype=np.float64),
        }
        h_blocks = {
            key0: np.asarray([[1.0e308]], dtype=np.float64),
            key1: np.asarray([[1.0e308]], dtype=np.float64),
        }
        with self.assertRaisesRegex(ValueError, "finite|overflow"):
            dotc(density, h_blocks)


class ExchangeContractionTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260811)
        self.C = rng.standard_normal((3, 4, 4)) + 1j * rng.standard_normal(
            (3, 4, 4)
        )
        factor = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        self.V = factor @ factor.conj().T
        self.O = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
        self.D = self.O @ self.O.conj().T

    def test_occupied_matches_direct_for_exact_density_factor(self) -> None:
        direct = direct_exchange(self.C, self.V, self.D)
        occupied = occupied_exchange(self.C, self.V, self.O)

        np.testing.assert_allclose(
            occupied.matrix, direct.matrix, atol=1.0e-12, rtol=0.0
        )
        self.assertIsInstance(direct, ExchangeResult)
        with self.assertRaises(FrozenInstanceError):
            direct.temporary_elements = 0

    def test_direct_matches_independent_explicit_loop(self) -> None:
        C = np.asarray(
            [
                [[1.0 + 0.5j, -0.25j], [0.75, -1.0j]],
                [[-0.5, 0.25 + 0.5j], [1.25j, 0.5]],
            ]
        )
        V = np.asarray([[2.0, 0.25j], [-0.25j, 1.5]])
        D = np.asarray([[1.25, 0.5 - 0.25j], [0.5 + 0.25j, 0.75]])
        expected = np.zeros((2, 2), dtype=np.complex128)
        for p in range(2):
            for s in range(2):
                for a in range(2):
                    for b in range(2):
                        for q in range(2):
                            for r in range(2):
                                expected[p, s] += (
                                    C[a, p, q]
                                    * D[q, r]
                                    * V[a, b]
                                    * C[b, s, r].conjugate()
                                )

        result = direct_exchange(C, V, D)

        np.testing.assert_allclose(result.matrix, expected, atol=1.0e-12, rtol=0.0)

    def test_zero_density_and_rank_zero_occupied_space_return_zero(self) -> None:
        direct = direct_exchange(self.C, self.V, np.zeros((4, 4)))
        occupied = occupied_exchange(self.C, self.V, np.empty((4, 0)))

        np.testing.assert_array_equal(direct.matrix, np.zeros((4, 4)))
        np.testing.assert_array_equal(occupied.matrix, np.zeros((4, 4)))
        self.assertEqual(occupied.temporary_shapes, ((3, 4, 0), (3, 4, 0), (4, 4)))
        self.assertEqual(occupied.temporary_elements, 16)

    def test_positive_semidefinite_metric_may_have_zero_modes(self) -> None:
        C = self.C[:2]
        V = np.diag([2.0, 0.0])

        direct = direct_exchange(C, V, self.D)
        occupied = occupied_exchange(C, V, self.O)

        np.testing.assert_allclose(
            occupied.matrix, direct.matrix, atol=1.0e-12, rtol=0.0
        )
        self.assertTrue(np.all(np.isfinite(direct.matrix)))

    def test_rejects_nonhermitian_metric_and_density(self) -> None:
        nonhermitian_V = self.V.copy()
        nonhermitian_V[0, 1] += 1.0e-4j
        nonhermitian_D = self.D.copy()
        nonhermitian_D[0, 1] += 1.0e-4

        for contraction in (
            lambda: direct_exchange(self.C, nonhermitian_V, self.D),
            lambda: direct_exchange(self.C, self.V, nonhermitian_D),
            lambda: occupied_exchange(self.C, nonhermitian_V, self.O),
        ):
            with self.subTest(contraction=contraction), self.assertRaisesRegex(
                ValueError, "Hermitian"
            ):
                contraction()

    def test_relative_hermiticity_check_is_scale_safe_and_does_not_symmetrize(self) -> None:
        huge_V = np.asarray(
            [[1.0e308 + 0.0j, 0.0], [0.0, -1.0e308 + 0.0j]]
        )
        zero_C = np.zeros((2, 2, 2), dtype=np.complex128)
        result = direct_exchange(zero_C, huge_V, np.eye(2))
        np.testing.assert_array_equal(result.matrix, np.zeros((2, 2)))

        C = np.eye(2, dtype=np.complex128)[None, :, :]
        raw_D = np.asarray([[1.0, 5.0e-13], [0.0, 1.0]], dtype=np.complex128)
        raw_result = direct_exchange(C, np.ones((1, 1)), raw_D)
        np.testing.assert_array_equal(raw_result.matrix, raw_D)

        near_V = np.asarray(
            [[1.0, 5.0e-13j], [0.0, 1.0]], dtype=np.complex128
        )
        scalar_result = occupied_exchange(
            np.ones((2, 1, 1)), near_V, np.ones((1, 1))
        )
        self.assertGreater(scalar_result.matrix[0, 0].imag, 0.0)

    def test_subnormal_nonhermitian_metric_is_rejected(self) -> None:
        tiny = np.nextafter(0.0, 1.0)
        V = np.asarray(
            [[tiny, tiny + 1j * tiny], [0.0, tiny]], dtype=np.complex128
        )

        with self.assertRaisesRegex(ValueError, "Hermitian"):
            occupied_exchange(np.ones((2, 1, 1)), V, np.ones((1, 1)))

    def test_subnormal_hermitian_metric_is_accepted_without_warning(self) -> None:
        tiny = np.nextafter(0.0, 1.0)
        V = np.asarray(
            [
                [tiny, tiny - 1j * tiny],
                [tiny + 1j * tiny, tiny],
            ],
            dtype=np.complex128,
        )

        result = occupied_exchange(
            np.zeros((2, 1, 1)), V, np.ones((1, 1))
        )

        self.assertTrue(np.all(np.isfinite(result.matrix)))

    def test_rejects_rank_shape_and_empty_dimension_errors(self) -> None:
        invalid_direct = (
            (np.ones((3, 4)), self.V, self.D),
            (np.ones((3, 4, 5)), self.V, self.D),
            (self.C, np.ones((3, 2)), self.D),
            (self.C, np.eye(2), self.D),
            (self.C, self.V, np.ones((4, 3))),
            (np.empty((0, 4, 4)), np.empty((0, 0)), self.D),
            (np.empty((2, 0, 0)), np.eye(2), np.empty((0, 0))),
        )
        for arguments in invalid_direct:
            with self.subTest(arguments=tuple(a.shape for a in arguments)):
                with self.assertRaises(ValueError):
                    direct_exchange(*arguments)

        invalid_occupied = (
            np.ones(4),
            np.ones((3, 2)),
            np.empty((4, 0, 1)),
        )
        for O in invalid_occupied:
            with self.subTest(shape=O.shape), self.assertRaises(ValueError):
                occupied_exchange(self.C, self.V, O)

    def test_rejects_failed_conversions_and_nonfinite_inputs(self) -> None:
        class CannotConvert:
            def __complex__(self):
                raise TypeError("conversion refused")

        with self.assertRaisesRegex(ValueError, "convert"):
            direct_exchange(CannotConvert(), self.V, self.D)
        with self.assertRaisesRegex(ValueError, "convert"):
            occupied_exchange(self.C, self.V, CannotConvert())

        nonfinite_V = self.V.copy()
        nonfinite_V[0, 0] = np.inf
        nonfinite_D = self.D.copy()
        nonfinite_D[0, 0] = np.nan
        nonfinite_O = self.O.copy()
        nonfinite_O[0, 0] = np.inf
        calls = (
            lambda: direct_exchange(
                np.where(np.indices(self.C.shape)[0] == 0, np.nan, self.C),
                self.V,
                self.D,
            ),
            lambda: direct_exchange(self.C, nonfinite_V, self.D),
            lambda: direct_exchange(self.C, self.V, nonfinite_D),
            lambda: occupied_exchange(self.C, self.V, nonfinite_O),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(ValueError, "finite"):
                call()

    def test_large_finite_contractions_that_overflow_raise_value_error(self) -> None:
        C = np.asarray([[[1.0e308 + 0.0j]]])
        V = np.ones((1, 1))

        for contraction in (
            lambda: direct_exchange(C, V, np.ones((1, 1))),
            lambda: occupied_exchange(C, V, np.ones((1, 1))),
        ):
            with self.subTest(contraction=contraction), self.assertRaisesRegex(
                ValueError, "finite|overflow"
            ):
                contraction()

    def test_inputs_are_c_contiguous_complex128_and_einsums_are_exact(self) -> None:
        C = np.asfortranarray(np.arange(12.0).reshape(3, 2, 2))
        V = np.asfortranarray(np.eye(3))
        D = np.asfortranarray(np.eye(2))
        O = np.asfortranarray(np.arange(4.0).reshape(2, 2))
        real_einsum = np.einsum

        direct_calls = []

        def record_direct(subscripts, *operands, **kwargs):
            direct_calls.append((subscripts, operands, kwargs))
            return real_einsum(subscripts, *operands, **kwargs)

        with mock.patch("exx_thc.supercell.np.einsum", side_effect=record_direct):
            direct = direct_exchange(C, V, D)
        self.assertEqual(
            [call[0] for call in direct_calls],
            ["apq,qr->apr", "ab,bsr->asr", "apr,asr->ps"],
        )
        for array in (
            direct_calls[0][1][0],
            direct_calls[0][1][1],
            direct_calls[1][1][0],
        ):
            self.assertEqual(array.dtype, np.dtype(np.complex128))
            self.assertTrue(array.flags.c_contiguous)
        self.assertTrue(all(call[2] == {"optimize": True} for call in direct_calls))
        self.assertEqual(direct.temporary_shapes, ((3, 2, 2), (3, 2, 2), (2, 2)))
        self.assertEqual(direct.temporary_elements, 28)

        occupied_calls = []

        def record_occupied(subscripts, *operands, **kwargs):
            occupied_calls.append((subscripts, operands, kwargs))
            return real_einsum(subscripts, *operands, **kwargs)

        with mock.patch("exx_thc.supercell.np.einsum", side_effect=record_occupied):
            occupied = occupied_exchange(C, V, O)
        self.assertEqual(
            [call[0] for call in occupied_calls],
            ["apq,qv->apv", "ab,bsv->asv", "apv,asv->ps"],
        )
        for array in (
            occupied_calls[0][1][0],
            occupied_calls[0][1][1],
            occupied_calls[1][1][0],
        ):
            self.assertEqual(array.dtype, np.dtype(np.complex128))
            self.assertTrue(array.flags.c_contiguous)
        self.assertTrue(all(call[2] == {"optimize": True} for call in occupied_calls))
        self.assertEqual(occupied.temporary_shapes, ((3, 2, 2), (3, 2, 2), (2, 2)))
        self.assertEqual(occupied.temporary_elements, 28)

    def test_reported_temporaries_are_never_four_index(self) -> None:
        for result in (
            direct_exchange(self.C, self.V, self.D),
            occupied_exchange(self.C, self.V, self.O),
        ):
            with self.subTest(result=result):
                self.assertTrue(all(len(shape) <= 3 for shape in result.temporary_shapes))
                self.assertEqual(
                    result.temporary_elements,
                    sum(int(np.prod(shape)) for shape in result.temporary_shapes),
                )


if __name__ == "__main__":
    unittest.main()
