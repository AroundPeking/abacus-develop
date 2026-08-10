import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from exx_thc.bvk import assemble_matrix, from_k, k_sector, to_k
from exx_thc.io import BlockKey, Snapshot, read_snapshot, write_snapshot


def _snapshot(blocks, *, rank=0, nranks=1, scalar="complex128"):
    return Snapshot(version=1, scalar=scalar, rank=rank, nranks=nranks, blocks=blocks)


def _run_cli(*arguments):
    environment = os.environ.copy()
    command = [sys.executable, "-m", "exx_thc.cli"] + [str(value) for value in arguments]
    return subprocess.run(command, text=True, capture_output=True, env=environment, check=False)


class CompareCommandTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "H.reference.exxcmp"
        self.candidate = self.root / "H.candidate.exxcmp"
        self.energy_reference = self.root / "E.reference.scalar"
        self.energy_candidate = self.root / "E.candidate.scalar"
        block = np.asarray([[1.0 + 0.5j, -0.25j]], dtype=np.complex128)
        blocks = {BlockKey(0, 0, (0, 0, 0)): block}
        write_snapshot(self.reference, _snapshot(blocks))
        write_snapshot(self.candidate, _snapshot({key: value.copy() for key, value in blocks.items()}))
        self.energy_reference.write_text("-2.0 0.0\n", encoding="utf-8")
        self.energy_candidate.write_text("-2.0 0.0\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def compare(self, *extra):
        return _run_cli(
            "compare",
            "--reference",
            self.reference,
            "--candidate",
            self.candidate,
            "--energy-reference",
            self.energy_reference,
            "--energy-candidate",
            self.energy_candidate,
            *extra,
        )

    def test_identical_snapshots_pass_and_infer_nat(self):
        result = self.compare()
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["nat"], 1)
        self.assertEqual(report["H_rel_fro"], 0.0)
        self.assertEqual(report["E_abs_Ry_atom"], 0.0)
        self.assertTrue(report["pass"])

    def test_identical_real_snapshots_pass(self):
        block = np.asarray([[1.0, -0.25]], dtype=np.float64)
        blocks = {BlockKey(0, 0, (0, 0, 0)): block}
        write_snapshot(self.reference, _snapshot(blocks, scalar="real64"))
        write_snapshot(self.candidate, _snapshot(blocks, scalar="real64"))

        result = self.compare()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(json.loads(result.stdout)["pass"])

    def test_tolerance_failure_returns_nonzero_json(self):
        changed = read_snapshot(self.candidate).blocks
        changed[BlockKey(0, 0, (0, 0, 0))][0, 0] += 1.0e-5
        write_snapshot(self.candidate, _snapshot(changed))
        result = self.compare("--nat", 2, "--h-rel-tol", 1.0e-12)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["nat"], 2)
        self.assertGreater(report["H_rel_fro"], 1.0e-12)
        self.assertFalse(report["pass"])

    def test_key_and_shape_sets_must_match_exactly(self):
        blocks = read_snapshot(self.candidate).blocks
        blocks[BlockKey(0, 0, (-1, 0, 0))] = np.ones((1, 2), dtype=np.complex128)
        write_snapshot(self.candidate, _snapshot(blocks))
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("key set", result.stderr)

    def test_energy_imaginary_part_at_gate_is_rejected(self):
        self.energy_candidate.write_text("-2.0 1e-13\n", encoding="utf-8")
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("imaginary", result.stderr)

    def test_nat_inference_rejects_noncontiguous_atom_indices(self):
        block = np.ones((1, 1), dtype=np.complex128)
        write_snapshot(
            self.reference,
            _snapshot({BlockKey(0, 2, (0, 0, 0)): block}),
        )
        write_snapshot(
            self.candidate,
            _snapshot({BlockKey(0, 2, (0, 0, 0)): block}),
        )
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("infer", result.stderr)

    def test_rejects_nonfinite_h_blocks(self):
        blocks = read_snapshot(self.candidate).blocks
        blocks[BlockKey(0, 0, (0, 0, 0))][0, 0] = np.nan + 0.0j
        write_snapshot(self.candidate, _snapshot(blocks))
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("finite", result.stderr)


class ProjectCommandTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.c_path = self.root / "C.exxcmp"
        self.d_path = self.root / "D.full.exxcmp"
        self.output = self.root / "C.occ.exxcmp"
        self.period = (2, 1, 1)

    def tearDown(self):
        self.temporary.cleanup()

    def run_project(self, *extra):
        return _run_cli(
            "project",
            "--C",
            self.c_path,
            "--D-full",
            self.d_path,
            "--period",
            *self.period,
            "--output",
            self.output,
            *extra,
        )

    def write_valid_case(self, *, full_rank=False):
        c_k = {
            (0, 0): np.asarray(
                [
                    [[[[1.0 + 0.2j, 0.4 - 0.1j], [0.3j, -0.8 + 0.1j]]]],
                    [[[[0.6 - 0.3j, -0.2j], [0.7 + 0.4j, 0.2]]]],
                ],
                dtype=np.complex128,
            ).reshape(self.period + (1, 2, 2))
        }
        if full_rank:
            d_sectors = [np.diag([0.7, 0.2]), np.diag([0.4, 0.1])]
        else:
            vector0 = np.asarray([1.0, 0.4j], dtype=np.complex128)
            vector1 = np.asarray([0.5 + 0.2j, -0.3j], dtype=np.complex128)
            d_sectors = [np.outer(vector0, vector0.conj()), np.outer(vector1, vector1.conj())]
        d_k = {(0, 0): np.asarray(d_sectors, dtype=np.complex128).reshape(self.period + (2, 2))}
        write_snapshot(self.c_path, _snapshot(from_k(c_k, self.period)))
        write_snapshot(self.d_path, _snapshot(from_k(d_k, self.period)))
        return c_k, d_k

    def write_sparse_two_atom_case(self):
        c_k = {
            (0, 0): np.asarray([1.0 + 0.2j, 0.7 - 0.1j], dtype=np.complex128).reshape(
                self.period + (1, 1, 1)
            ),
            (1, 1): np.asarray([0.4 - 0.3j, -0.2 + 0.6j], dtype=np.complex128).reshape(
                self.period + (1, 1, 1)
            ),
        }
        density_sectors = []
        for vector in (
            np.asarray([1.0, 0.5j], dtype=np.complex128),
            np.asarray([0.3 + 0.4j, -0.6j], dtype=np.complex128),
        ):
            density_sectors.append(np.outer(vector, vector.conj()))
        d_k = {}
        for ia1 in range(2):
            for ia2 in range(2):
                d_k[(ia1, ia2)] = np.asarray(
                    [density[ia1, ia2] for density in density_sectors],
                    dtype=np.complex128,
                ).reshape(self.period + (1, 1))
        write_snapshot(self.c_path, _snapshot(from_k(c_k, self.period)))
        write_snapshot(self.d_path, _snapshot(from_k(d_k, self.period)))
        return c_k, d_k

    def write_small_scale_density_case(self, negative_eigenvalue, positive_eigenvalue=1.0e-6):
        c_k = {
            (0, 0): np.asarray(
                [
                    [[[[1.0, 0.2j], [0.3, -0.4j]]]],
                    [[[[0.7j, -0.1], [0.5, 0.6j]]]],
                ],
                dtype=np.complex128,
            ).reshape(self.period + (1, 2, 2))
        }
        density = np.diag([positive_eigenvalue, negative_eigenvalue]).astype(np.complex128)
        d_k = {
            (0, 0): np.asarray([density, density], dtype=np.complex128).reshape(
                self.period + (2, 2)
            )
        }
        write_snapshot(self.c_path, _snapshot(from_k(c_k, self.period)))
        write_snapshot(self.d_path, _snapshot(from_k(d_k, self.period)))

    def write_large_c_roundtrip_case(self):
        coefficient = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[[1.0e4 + 0.3j]]], dtype=np.complex128),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[[0.7 - 0.2j]]], dtype=np.complex128),
        }
        density = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[1.0]], dtype=np.complex128)
        }
        write_snapshot(self.c_path, _snapshot(coefficient))
        write_snapshot(self.d_path, _snapshot(density))

    def test_project_preserves_cdc_in_every_k_sector(self):
        c_k, d_k = self.write_valid_case()
        result = self.run_project()
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertLess(report["fourier_roundtrip_max"], 1.0e-13)
        self.assertLess(report["D_hermiticity_max"], 1.0e-12)
        self.assertEqual(report["occupied_ranks_by_k"], [1, 1])
        self.assertGreater(report["discarded_trace"], -1.0e-15)

        projected = read_snapshot(self.output)
        self.assertEqual(projected.scalar, "complex128")
        self.assertEqual((projected.rank, projected.nranks), (0, 1))
        expected_keys = {
            BlockKey(0, 0, (0, 0, 0)),
            BlockKey(0, 0, (-1, 0, 0)),
        }
        self.assertEqual(set(projected.blocks), expected_keys)
        projected_k = to_k(projected.blocks, self.period)
        for k_index in np.ndindex(self.period):
            c_original = np.asarray(c_k[(0, 0)][k_index]).reshape(2, 2)
            c_candidate = np.asarray(projected_k[(0, 0)][k_index]).reshape(2, 2)
            density = assemble_matrix(k_sector(d_k, k_index))
            reference_product = c_original @ density @ c_original.conj().T
            candidate_product = c_candidate @ density @ c_candidate.conj().T
            np.testing.assert_allclose(candidate_product, reference_product, rtol=0.0, atol=1.0e-12)

    def test_full_rank_projection_preserves_c_and_dtype(self):
        self.write_valid_case(full_rank=True)
        original = read_snapshot(self.c_path)
        result = self.run_project()
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        projected = read_snapshot(self.output)
        self.assertEqual(projected.scalar, "complex128")
        self.assertEqual(set(projected.blocks), set(original.blocks))
        for key in original.blocks:
            np.testing.assert_allclose(projected.blocks[key], original.blocks[key], rtol=0.0, atol=1.0e-13)

    def test_real_snapshots_produce_real_projected_snapshot(self):
        self.period = (1, 1, 1)
        coefficient = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray(
                [[[1.0, 0.2], [0.3, -0.4]]], dtype=np.float64
            )
        }
        density = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray(
                [[1.0, 0.0], [0.0, 0.0]], dtype=np.float64
            )
        }
        write_snapshot(self.c_path, _snapshot(coefficient, scalar="real64"))
        write_snapshot(self.d_path, _snapshot(density, scalar="real64"))

        result = self.run_project()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        projected = read_snapshot(self.output)
        self.assertEqual(projected.scalar, "real64")
        self.assertTrue(all(block.dtype == np.float64 for block in projected.blocks.values()))

    def test_projection_emits_cross_atom_pairs_created_by_global_projector(self):
        c_k, d_k = self.write_sparse_two_atom_case()
        result = self.run_project()
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        projected = read_snapshot(self.output)
        expected_keys = {
            BlockKey(ia1, ia2, R)
            for ia1 in range(2)
            for ia2 in range(2)
            for R in ((0, 0, 0), (-1, 0, 0))
        }
        self.assertEqual(set(projected.blocks), expected_keys)
        projected_k = to_k(projected.blocks, self.period)
        for k_index in np.ndindex(self.period):
            c_original = np.diag(
                [c_k[(0, 0)][k_index].item(), c_k[(1, 1)][k_index].item()]
            ).astype(np.complex128)
            c_candidate = np.asarray(
                [
                    [projected_k[(0, 0)][k_index].item(), projected_k[(0, 1)][k_index].item()],
                    [projected_k[(1, 0)][k_index].item(), projected_k[(1, 1)][k_index].item()],
                ],
                dtype=np.complex128,
            )
            density = assemble_matrix(k_sector(d_k, k_index))
            reference_product = c_original @ density @ c_original.conj().T
            candidate_product = c_candidate @ density @ c_candidate.conj().T
            np.testing.assert_allclose(candidate_product, reference_product, rtol=0.0, atol=1.0e-12)

    def test_psd_gate_scales_negative_floor_by_positive_lambda_max(self):
        self.write_small_scale_density_case(-5.0e-11)
        result = self.run_project()
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertFalse(report["pass"])
        self.assertAlmostEqual(report["D_min_eigenvalue_scaled"], -5.0e-5)
        self.assertFalse(self.output.exists())

    def test_psd_gate_accepts_zero_density_when_lambda_max_is_zero(self):
        self.write_small_scale_density_case(0.0, positive_eigenvalue=0.0)
        result = self.run_project()
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["pass"])
        self.assertEqual(report["D_min_eigenvalue_scaled"], 0.0)

    def test_fourier_gate_checks_c_and_d_roundtrips_separately(self):
        self.write_large_c_roundtrip_case()
        coefficient_k = to_k(read_snapshot(self.c_path).blocks, self.period)
        coefficient_roundtrip_k = to_k(from_k(coefficient_k, self.period), self.period)
        expected_c_error = max(
            float(np.max(np.abs(coefficient_k[pair] - coefficient_roundtrip_k[pair])))
            for pair in coefficient_k
        )
        self.assertGreater(expected_c_error, 1.0e-13)

        result = self.run_project()
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertAlmostEqual(report["C_fourier_roundtrip_max"], expected_c_error)
        self.assertLess(report["D_fourier_roundtrip_max"], 1.0e-13)
        self.assertEqual(
            report["fourier_roundtrip_max"],
            max(report["C_fourier_roundtrip_max"], report["D_fourier_roundtrip_max"]),
        )
        self.assertFalse(report["pass"])
        self.assertEqual(report["output_blocks"], 0)
        self.assertEqual(report["output_bytes_estimate"], 0)
        self.assertFalse(self.output.exists())

    def test_rejects_nonfinite_c_without_writing_output(self):
        self.write_valid_case()
        snapshot = read_snapshot(self.c_path)
        next(iter(snapshot.blocks.values())).flat[0] = np.nan + 0.0j
        write_snapshot(self.c_path, snapshot)
        result = self.run_project()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("finite", result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_fourier_alias_sum_overflow_without_writing_output(self):
        self.write_valid_case()
        shape = next(iter(read_snapshot(self.c_path).blocks.values())).shape
        coefficient = {
            BlockKey(0, 0, (0, 0, 0)): np.full(shape, 1.0e308, dtype=np.complex128),
            BlockKey(0, 0, (2, 0, 0)): np.full(shape, 1.0e308, dtype=np.complex128),
        }
        write_snapshot(self.c_path, _snapshot(coefficient))
        result = self.run_project()
        self.assertEqual(result.returncode, 2)
        self.assertIn("finite", result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_distributed_snapshots(self):
        self.write_valid_case()
        snapshot = read_snapshot(self.c_path)
        write_snapshot(
            self.c_path,
            Snapshot(1, snapshot.scalar, 0, 2, snapshot.blocks),
        )
        result = self.run_project()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank 0 of 1", result.stderr)

    def test_rejects_malformed_c_dimensions_without_output(self):
        self.write_valid_case()
        malformed = {BlockKey(0, 0, (0, 0, 0)): np.ones((2, 2), dtype=np.complex128)}
        write_snapshot(self.c_path, _snapshot(malformed))
        result = self.run_project()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("three-dimensional", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
