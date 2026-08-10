from __future__ import annotations

import itertools
import unittest

import numpy as np

from exx_thc.bvk import assemble_matrix, from_k, k_sector, to_k
from exx_thc.io import BlockKey


class BvKTransformTest(unittest.TestCase):
    def test_roundtrip_uses_negative_canonical_representatives(self) -> None:
        rng = np.random.default_rng(718)
        period = (2, 2, 1)
        blocks = {}
        for r in itertools.product((0, -1), (0, -1), (0,)):
            blocks[BlockKey(0, 1, r)] = (
                rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
            ).astype(np.complex128)

        restored = from_k(to_k(blocks, period), period)

        self.assertEqual(set(restored), set(blocks))
        self.assertTrue(any(-1 in key.R for key in restored))
        for key, expected in blocks.items():
            np.testing.assert_allclose(restored[key], expected, rtol=0.0, atol=1e-13)
            self.assertEqual(restored[key].dtype, np.dtype(np.complex128))

    def test_alias_translations_are_summed_and_missing_r_is_zero(self) -> None:
        blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.array([[2.0]], dtype=np.float64),
            BlockKey(0, 0, (2, 0, 0)): np.array([[3.0]], dtype=np.float64),
        }

        transformed = to_k(blocks, (2, 1, 1))

        np.testing.assert_array_equal(transformed[(0, 0)][..., 0, 0], np.array([[[5.0]], [[5.0]]]))
        restored = from_k(transformed, (2, 1, 1))
        np.testing.assert_allclose(
            restored[BlockKey(0, 0, (0, 0, 0))], [[5.0]], rtol=0.0, atol=1e-13
        )
        np.testing.assert_allclose(
            restored[BlockKey(0, 0, (-1, 0, 0))], [[0.0]], rtol=0.0, atol=1e-13
        )

    def test_hermitian_real_space_relation_produces_hermitian_k_matrices(self) -> None:
        rng = np.random.default_rng(921)
        period = (3, 2, 1)
        blocks = {}
        translations = list(itertools.product((0, 1, -1), (0, -1), (0,)))

        def canonical(value: int, extent: int) -> int:
            residue = value % extent
            return residue if residue <= (extent - 1) // 2 else residue - extent

        def inverse(cell: tuple[int, int, int]) -> tuple[int, int, int]:
            return tuple(canonical(-value, extent) for value, extent in zip(cell, period))

        for atom, size in ((0, 2), (1, 1)):
            for r in translations:
                key = BlockKey(atom, atom, r)
                if key in blocks:
                    continue
                inverse_r = inverse(r)
                value = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
                if r == inverse_r:
                    value = value + value.conj().T
                blocks[key] = value.astype(np.complex128)
                blocks[BlockKey(atom, atom, inverse_r)] = value.conj().T.astype(np.complex128)

        for r in translations:
            d01 = (rng.normal(size=(2, 1)) + 1j * rng.normal(size=(2, 1))).astype(
                np.complex128
            )
            blocks[BlockKey(0, 1, r)] = d01
            blocks[BlockKey(1, 0, inverse(r))] = d01.conj().T

        transformed = to_k(blocks, period)
        for k_index in itertools.product(range(3), range(2), range(1)):
            matrix = assemble_matrix(k_sector(transformed, k_index))
            np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-13)

    def test_assemble_matrix_sorts_atoms_and_fills_missing_pairs(self) -> None:
        blocks = {
            (2, 2): np.array([[4.0]], dtype=np.float64),
            (0, 0): np.eye(2, dtype=np.float64),
            (0, 2): np.array([[1.0], [2.0]], dtype=np.float64),
        }

        matrix = assemble_matrix(blocks)

        expected = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 4.0]],
            dtype=np.complex128,
        )
        np.testing.assert_array_equal(matrix, expected)
        self.assertEqual(matrix.dtype, np.dtype(np.complex128))

    def test_rejects_period_shape_and_atom_dimension_conflicts(self) -> None:
        with self.assertRaises(ValueError):
            to_k({}, (2, 0, 1))
        with self.assertRaises(ValueError):
            to_k({}, (2, 2))
        with self.assertRaises(ValueError):
            to_k(
                {
                    BlockKey(0, 1, (0, 0, 0)): np.ones((2, 3)),
                    BlockKey(0, 1, (-1, 0, 0)): np.ones((2, 4)),
                },
                (2, 1, 1),
            )
        with self.assertRaises(ValueError):
            assemble_matrix(
                {
                    (0, 1): np.ones((2, 3)),
                    (1, 0): np.ones((4, 2)),
                }
            )


if __name__ == "__main__":
    unittest.main()
