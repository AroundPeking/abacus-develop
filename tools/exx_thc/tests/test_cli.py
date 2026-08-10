import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from exx_thc import cli
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


class SupercellGateCommandTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.c_path = self.root / "C.active.exxcmp"
        self.v_path = self.root / "V.active.exxcmp"
        self.d_full_path = self.root / "D.full.exxcmp"
        self.d_post_path = self.root / "D.raw.exxcmp"
        self.h_reference_path = self.root / "H.lri.exxcmp"
        self.energy_reference_path = self.root / "E.lri.scalar"
        self.h_dense_out = self.root / "H.dense.exxcmp"
        self.h_occ_out = self.root / "H.occ.exxcmp"
        self.period = (1, 1, 1)
        self.max_elements = 1000

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _explicit_exchange(coefficient, metric, density):
        naux, nao, _ = coefficient.shape
        result = np.zeros((nao, nao), dtype=np.complex128)
        for p in range(nao):
            for s in range(nao):
                for a in range(naux):
                    for b in range(naux):
                        for q in range(nao):
                            for r in range(nao):
                                result[p, s] += (
                                    coefficient[a, p, q]
                                    * density[q, r]
                                    * metric[a, b]
                                    * coefficient[b, s, r].conjugate()
                                )
        return result

    @staticmethod
    def _explicit_dotc(density, hamiltonian):
        result = 0.0j
        for row in range(density.shape[0]):
            for column in range(density.shape[1]):
                result += (
                    density[row, column].conjugate()
                    * hamiltonian[row, column]
                )
        return result

    def write_valid_case(self, *, scalar="real64", single_auxiliary=False):
        if scalar == "real64":
            coefficient = np.asarray(
                [
                    [[0.8, -0.2], [-0.2, 0.5]],
                    [[0.3, 0.4], [0.4, -0.6]],
                ],
                dtype=np.float64,
            )
            metric = np.asarray([[1.2, 0.15], [0.15, 0.9]], dtype=np.float64)
            density = np.asarray([[0.8, 0.1], [0.1, 0.3]], dtype=np.float64)
            density_post = np.asarray([[0.4, -0.05], [-0.05, 0.2]], dtype=np.float64)
        else:
            coefficient = np.asarray(
                [
                    [[0.8 + 0.1j, -0.2 + 0.3j], [-0.2 + 0.3j, 0.5 - 0.2j]],
                    [[0.3 - 0.1j, 0.4 + 0.2j], [0.4 + 0.2j, -0.6 + 0.1j]],
                ],
                dtype=np.complex128,
            )
            metric = np.asarray(
                [[1.2, 0.15 + 0.05j], [0.15 - 0.05j, 0.9]],
                dtype=np.complex128,
            )
            density = np.asarray(
                [[0.8, 0.1 + 0.07j], [0.1 - 0.07j, 0.3]],
                dtype=np.complex128,
            )
            density_post = np.asarray(
                [[0.4, -0.05 + 0.02j], [-0.05 - 0.02j, 0.2]],
                dtype=np.complex128,
            )

        if single_auxiliary:
            coefficient = coefficient[:1]
            metric = metric[:1, :1]

        dtype = np.float64 if scalar == "real64" else np.complex128
        key = BlockKey(0, 0, (0, 0, 0))
        expected_h = self._explicit_exchange(coefficient, metric, density)
        stored_h = np.asarray(
            expected_h.real if scalar == "real64" else expected_h,
            dtype=dtype,
            order="C",
        )
        write_snapshot(
            self.c_path,
            _snapshot({key: np.asarray(0.5 * coefficient, dtype=dtype)}, scalar=scalar),
        )
        write_snapshot(self.v_path, _snapshot({key: metric}, scalar=scalar))
        write_snapshot(self.d_full_path, _snapshot({key: density}, scalar=scalar))
        write_snapshot(self.d_post_path, _snapshot({key: density_post}, scalar=scalar))
        write_snapshot(self.h_reference_path, _snapshot({key: stored_h}, scalar=scalar))
        energy = self._explicit_dotc(density_post, stored_h)
        self.energy_reference_path.write_text(
            "{:.17e} {:.17e}\n".format(float(energy.real), float(energy.imag)),
            encoding="utf-8",
        )
        return key, stored_h, density_post

    def gate_arguments(self, *extra, include_max_elements=True):
        arguments = [
            "supercell-gate",
            "--C",
            self.c_path,
            "--V",
            self.v_path,
            "--D-full",
            self.d_full_path,
            "--D-post",
            self.d_post_path,
            "--H-reference",
            self.h_reference_path,
            "--energy-reference",
            self.energy_reference_path,
            "--period",
            *self.period,
            "--H-dense-out",
            self.h_dense_out,
            "--H-occ-out",
            self.h_occ_out,
        ]
        if include_max_elements:
            arguments.extend(("--max-elements", self.max_elements))
        arguments.extend(extra)
        return arguments

    def run_gate(self, *extra, include_max_elements=True):
        return _run_cli(
            *self.gate_arguments(*extra, include_max_elements=include_max_elements)
        )

    def snapshot_elements_upper_bound(self):
        return sum(
            (path.stat().st_size + 15) // 16
            for path in (
                self.c_path,
                self.v_path,
                self.d_full_path,
                self.d_post_path,
                self.h_reference_path,
            )
        )

    def expected_live_upper_bound(self, *, naux, nao, scalar):
        coefficient = naux * nao * nao
        metric = naux * naux
        density = nao * nao
        factor_peak = coefficient + metric + 10 * density
        if scalar == "real64":
            contraction_peak = 4 * coefficient + 2 * metric + 4 * density
        else:
            direct_peak = 3 * coefficient + metric + 3 * density
            occupied_peak = 3 * coefficient + metric + 4 * density
            contraction_peak = max(direct_peak, occupied_peak)
        return self.snapshot_elements_upper_bound() + max(
            factor_peak, contraction_peak
        )

    def remove_outputs(self):
        for output in (self.h_dense_out, self.h_occ_out):
            try:
                output.unlink()
            except FileNotFoundError:
                pass

    def assert_no_outputs(self):
        self.assertFalse(self.h_dense_out.exists())
        self.assertFalse(self.h_occ_out.exists())

    def test_exact_real_case_publishes_both_outputs_and_complete_report(self):
        key, expected_h, _density_post = self.write_valid_case()

        result = self.run_gate()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["pass"])
        self.assertEqual(report["nat"], 1)
        self.assertEqual(report["occupied_ranks_by_supercell"], [2])
        self.assertEqual(report["dense_bytes"], 16 * 20)
        self.assertEqual(report["occupied_bytes"], 16 * 20)
        self.assertEqual(
            report["live_elements_upper_bound"],
            self.expected_live_upper_bound(naux=2, nao=2, scalar="real64"),
        )
        self.assertEqual(
            report["snapshot_elements_upper_bound"],
            self.snapshot_elements_upper_bound(),
        )
        self.assertEqual(report["max_elements"], self.max_elements)
        for field in (
            "dense_H_rel_fro",
            "occupied_H_rel_fro",
            "dense_occ_H_rel_fro",
            "dense_E_abs_Ry_atom",
            "occupied_E_abs_Ry_atom",
            "D_hermiticity_rel_fro",
            "density_factor_rel_fro",
        ):
            self.assertLessEqual(report[field], 1.0e-12)
        self.assertGreaterEqual(report["discarded_trace"], 0.0)
        self.assertGreater(report["D_min_eigenvalue"], 0.0)
        self.assertGreater(report["D_min_eigenvalue_scaled"], 0.0)
        for output in (self.h_dense_out, self.h_occ_out):
            snapshot = read_snapshot(output)
            self.assertEqual(snapshot.scalar, "real64")
            self.assertEqual((snapshot.rank, snapshot.nranks), (0, 1))
            self.assertEqual(set(snapshot.blocks), {key})
            np.testing.assert_allclose(snapshot.blocks[key], expected_h, rtol=0.0, atol=1.0e-12)

    def test_h_reference_mismatch_is_metric_failure_without_outputs(self):
        key, expected_h, _density_post = self.write_valid_case()
        changed = expected_h.copy()
        changed[0, 0] += 1.0e-4
        write_snapshot(self.h_reference_path, _snapshot({key: changed}, scalar="real64"))

        result = self.run_gate()

        self.assertEqual(result.returncode, 1, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["pass"])
        self.assertGreater(report["dense_H_rel_fro"], 1.0e-10)
        self.assertGreater(report["occupied_H_rel_fro"], 1.0e-10)
        self.assert_no_outputs()

    def test_energy_reference_mismatch_is_metric_failure_without_outputs(self):
        self.write_valid_case()
        real, imaginary = (
            float(value)
            for value in self.energy_reference_path.read_text(
                encoding="utf-8"
            ).split()
        )
        self.energy_reference_path.write_text(
            "{:.17e} {:.17e}\n".format(real + 1.0e-4, imaginary),
            encoding="utf-8",
        )

        result = self.run_gate()

        self.assertEqual(result.returncode, 1, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["pass"])
        self.assertLessEqual(report["dense_H_rel_fro"], 1.0e-10)
        self.assertLessEqual(report["occupied_H_rel_fro"], 1.0e-10)
        self.assertGreater(report["dense_E_abs_Ry_atom"], 1.0e-10)
        self.assertGreater(report["occupied_E_abs_Ry_atom"], 1.0e-10)
        self.assert_no_outputs()

    def test_reference_atom_universe_cannot_expand_nat_or_relax_energy_gate(self):
        _key, expected_h, _density_post = self.write_valid_case()
        blocks = read_snapshot(self.h_reference_path).blocks
        for atom in range(1, 10):
            blocks[BlockKey(atom, atom, (0, 0, 0))] = np.zeros_like(expected_h)
        write_snapshot(
            self.h_reference_path,
            _snapshot(blocks, scalar="real64"),
        )
        real, imaginary = (
            float(value)
            for value in self.energy_reference_path.read_text(
                encoding="utf-8"
            ).split()
        )
        self.energy_reference_path.write_text(
            "{:.17e} {:.17e}\n".format(real + 5.0e-10, imaginary),
            encoding="utf-8",
        )

        result = self.run_gate()

        self.assertEqual(result.returncode, 2, msg=result.stdout)
        self.assertIn("atom universe", result.stderr)
        self.assert_no_outputs()

    def test_dpost_cannot_contain_atoms_outside_layout(self):
        self.write_valid_case()
        density_post = read_snapshot(self.d_post_path).blocks
        density_post[BlockKey(99, 99, (0, 0, 0))] = np.ones(
            (2, 2), dtype=np.float64
        )
        write_snapshot(
            self.d_post_path,
            _snapshot(density_post, scalar="real64"),
        )

        result = self.run_gate()

        self.assertEqual(result.returncode, 2, msg=result.stdout)
        self.assertIn("atom universe", result.stderr)
        self.assert_no_outputs()

    def test_density_hermiticity_and_psd_failures_emit_json_without_outputs(self):
        key, _expected_h, _density_post = self.write_valid_case()
        nonhermitian = np.asarray([[0.8, 0.4], [0.1, 0.3]], dtype=np.float64)
        write_snapshot(self.d_full_path, _snapshot({key: nonhermitian}, scalar="real64"))

        result = self.run_gate()

        self.assertEqual(result.returncode, 1, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["pass"])
        self.assertGreater(report["D_hermiticity_rel_fro"], 1.0e-12)
        self.assertIsNone(report["dense_H_rel_fro"])
        self.assertIsNone(report["dense_E_abs_Ry_atom"])
        self.assert_no_outputs()

        self.write_valid_case()
        indefinite = np.asarray([[1.0, 0.0], [0.0, -1.0e-3]], dtype=np.float64)
        write_snapshot(self.d_full_path, _snapshot({key: indefinite}, scalar="real64"))

        result = self.run_gate()

        self.assertEqual(result.returncode, 1, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["pass"])
        self.assertLess(report["D_min_eigenvalue_scaled"], -1.0e-10)
        self.assertIsNone(report["dense_H_rel_fro"])
        self.assert_no_outputs()

    def test_memory_limit_failure_is_invalid_and_writes_nothing(self):
        self.write_valid_case()
        self.max_elements = 1

        result = self.run_gate()

        self.assertEqual(result.returncode, 2)
        self.assertIn("dense supercell live allocation exceeds max_elements", result.stderr)
        self.assert_no_outputs()

    def test_snapshot_payload_limit_is_checked_before_reading_snapshots(self):
        self.write_valid_case()
        self.max_elements = self.snapshot_elements_upper_bound() - 1
        arguments = cli._parser().parse_args(
            [str(value) for value in self.gate_arguments()]
        )

        with mock.patch.object(
            cli, "_serial_numeric_snapshot", wraps=cli._serial_numeric_snapshot
        ) as reader:
            with self.assertRaisesRegex(
                ValueError, "dense supercell live allocation exceeds max_elements"
            ):
                cli.supercell_gate(arguments)

        self.assertEqual(reader.call_count, 0)
        self.assert_no_outputs()

    def test_factor_phase_and_snapshot_payload_define_live_upper_bound(self):
        self.write_valid_case(single_auxiliary=True)

        result = self.run_gate()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        coefficient = 1 * 2 * 2
        metric = 1 * 1
        density = 2 * 2
        factor_peak = coefficient + metric + 10 * density
        contraction_peak = 4 * coefficient + 2 * metric + 4 * density
        self.assertGreater(factor_peak, contraction_peak)
        self.assertEqual(
            report["live_elements_upper_bound"],
            self.snapshot_elements_upper_bound() + factor_peak,
        )
        self.assertEqual(
            report["snapshot_elements_upper_bound"],
            self.snapshot_elements_upper_bound(),
        )

    def test_real_live_allocation_limit_includes_complex_conversion_copies(self):
        self.write_valid_case()

        initial = self.run_gate()

        self.assertEqual(initial.returncode, 0, msg=initial.stderr)
        initial_report = json.loads(initial.stdout)
        bound = initial_report["live_elements_upper_bound"]
        self.assertEqual(
            bound, self.expected_live_upper_bound(naux=2, nao=2, scalar="real64")
        )
        self.remove_outputs()
        self.max_elements = bound - 1

        rejected = self.run_gate()

        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            "dense supercell live allocation exceeds max_elements", rejected.stderr
        )
        self.assert_no_outputs()

        self.max_elements = bound
        accepted = self.run_gate()

        self.assertEqual(accepted.returncode, 0, msg=accepted.stderr)
        report = json.loads(accepted.stdout)
        self.assertEqual(report["live_elements_upper_bound"], bound)
        self.assertEqual(report["max_elements"], bound)

    def test_extreme_psd_failure_still_emits_strict_json(self):
        key, _expected_h, _density_post = self.write_valid_case()
        extreme = np.diag([-1.0e300, 1.0e-100]).astype(np.float64)
        write_snapshot(self.d_full_path, _snapshot({key: extreme}, scalar="real64"))

        result = self.run_gate()

        self.assertEqual(result.returncode, 1, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["pass"])
        self.assertLess(report["D_min_eigenvalue_scaled"], -1.0e-10)
        self.assert_no_outputs()

    def test_rejects_scalar_mismatch_rank_shard_and_invalid_period(self):
        key, _expected_h, _density_post = self.write_valid_case()
        metric = read_snapshot(self.v_path).blocks[key].astype(np.complex128)
        write_snapshot(self.v_path, _snapshot({key: metric}, scalar="complex128"))
        result = self.run_gate()
        self.assertEqual(result.returncode, 2)
        self.assertIn("scalar", result.stderr)
        self.assert_no_outputs()

        self.write_valid_case()
        coefficient = read_snapshot(self.c_path)
        write_snapshot(
            self.c_path,
            Snapshot(1, coefficient.scalar, 0, 2, coefficient.blocks),
        )
        result = self.run_gate()
        self.assertEqual(result.returncode, 2)
        self.assertIn("rank 0 of 1", result.stderr)
        self.assert_no_outputs()

        self.write_valid_case()
        self.period = (0, 1, 1)
        result = self.run_gate()
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive", result.stderr)
        self.assert_no_outputs()

    def test_max_elements_is_required_and_positive(self):
        self.write_valid_case()

        missing = self.run_gate(include_max_elements=False)
        zero = self.run_gate("--max-elements", 0, include_max_elements=False)

        self.assertEqual(missing.returncode, 2)
        self.assertEqual(zero.returncode, 2)
        self.assertIn("required", missing.stderr)
        self.assertIn("positive integer", zero.stderr)
        self.assert_no_outputs()

    def test_preexisting_output_is_never_overwritten(self):
        self.write_valid_case()
        sentinel = b"external output\n"
        for occupied_exists in (False, True):
            with self.subTest(occupied_exists=occupied_exists):
                existing = self.h_occ_out if occupied_exists else self.h_dense_out
                absent = self.h_dense_out if occupied_exists else self.h_occ_out
                existing.write_bytes(sentinel)

                result = self.run_gate()

                self.assertEqual(result.returncode, 2)
                self.assertIn("already exists", result.stderr)
                self.assertEqual(existing.read_bytes(), sentinel)
                self.assertFalse(absent.exists())
                existing.unlink()

    def test_complex128_exact_case_succeeds(self):
        key, expected_h, _density_post = self.write_valid_case(scalar="complex128")

        result = self.run_gate()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["pass"])
        bound = report["live_elements_upper_bound"]
        self.assertEqual(
            bound,
            self.expected_live_upper_bound(naux=2, nao=2, scalar="complex128"),
        )
        self.assertEqual(
            report["snapshot_elements_upper_bound"],
            self.snapshot_elements_upper_bound(),
        )
        for output in (self.h_dense_out, self.h_occ_out):
            snapshot = read_snapshot(output)
            self.assertEqual(snapshot.scalar, "complex128")
            np.testing.assert_allclose(snapshot.blocks[key], expected_h, rtol=0.0, atol=1.0e-12)

        self.remove_outputs()
        self.max_elements = bound - 1
        rejected = self.run_gate()
        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            "dense supercell live allocation exceeds max_elements", rejected.stderr
        )
        self.assert_no_outputs()

        self.max_elements = bound
        accepted = self.run_gate()
        self.assertEqual(accepted.returncode, 0, msg=accepted.stderr)

    def test_joint_publication_rolls_back_own_first_link_but_keeps_external_second(self):
        key = BlockKey(0, 0, (0, 0, 0))
        snapshot = _snapshot({key: np.ones((1, 1), dtype=np.float64)}, scalar="real64")
        real_link = os.link
        calls = []

        def fail_second_link(source, destination):
            calls.append((Path(source), Path(destination)))
            if Path(destination) == self.h_occ_out:
                Path(destination).write_bytes(b"external race\n")
                raise FileExistsError("simulated second-link race")
            real_link(source, destination)

        with mock.patch.object(os, "link", side_effect=fail_second_link):
            with self.assertRaises(FileExistsError):
                cli._publish_snapshots_together(
                    self.h_dense_out,
                    snapshot,
                    self.h_occ_out,
                    snapshot,
                )

        self.assertFalse(self.h_dense_out.exists())
        self.assertEqual(self.h_occ_out.read_bytes(), b"external race\n")
        self.assertEqual(list(self.root.glob(".supercell-gate*")), [])

    def test_joint_publication_rolls_back_both_links_when_stage_cleanup_fails(self):
        key = BlockKey(0, 0, (0, 0, 0))
        snapshot = _snapshot({key: np.ones((1, 1), dtype=np.float64)}, scalar="real64")
        real_rmtree = shutil.rmtree
        cleanup_failed = [False]

        def fail_first_stage_cleanup(path, *arguments, **keywords):
            if Path(path).name.startswith(
                (".supercell-gate.", ".supercell-gate-stage.")
            ) and not cleanup_failed[0]:
                cleanup_failed[0] = True
                raise OSError("simulated stage cleanup failure")
            return real_rmtree(path, *arguments, **keywords)

        with mock.patch.object(os, "link", wraps=os.link) as link:
            with mock.patch.object(
                cli, "_fsync_directory", wraps=cli._fsync_directory
            ) as fsync:
                with mock.patch.object(
                    shutil, "rmtree", side_effect=fail_first_stage_cleanup
                ):
                    with self.assertRaisesRegex(
                        OSError, "simulated stage cleanup failure"
                    ):
                        cli._publish_snapshots_together(
                            self.h_dense_out,
                            snapshot,
                            self.h_occ_out,
                            snapshot,
                        )

        final_links = [
            call
            for call in link.call_args_list
            if Path(call.args[1]) in (self.h_dense_out, self.h_occ_out)
        ]
        self.assertEqual(len(final_links), 2)
        self.assertGreaterEqual(fsync.call_count, 2)
        self.assert_no_outputs()
        self.assertEqual(list(self.root.glob(".supercell-gate*")), [])

    def test_joint_publication_does_not_open_stage_snapshot_files(self):
        key = BlockKey(0, 0, (0, 0, 0))
        snapshot = _snapshot({key: np.ones((1, 1), dtype=np.float64)}, scalar="real64")
        real_open = os.open
        opened_paths = []

        def reject_stage_snapshot_open(path, *arguments, **keywords):
            opened_path = Path(path)
            opened_paths.append(opened_path)
            if opened_path.name.endswith((".stage.exxcmp", ".owner.exxcmp")):
                raise AssertionError("publication opened an ownership-linked file")
            return real_open(path, *arguments, **keywords)

        with mock.patch.object(os, "open", side_effect=reject_stage_snapshot_open):
            cli._publish_snapshots_together(
                self.h_dense_out,
                snapshot,
                self.h_occ_out,
                snapshot,
            )

        self.assertTrue(opened_paths)
        for output in (self.h_dense_out, self.h_occ_out):
            published = read_snapshot(output)
            np.testing.assert_array_equal(published.blocks[key], np.ones((1, 1)))

    def test_joint_publication_uses_owner_anchors_after_partial_stage_cleanup(self):
        key = BlockKey(0, 0, (0, 0, 0))
        snapshot = _snapshot({key: np.ones((1, 1), dtype=np.float64)}, scalar="real64")
        real_rmtree = shutil.rmtree
        stage_cleanup_failed = [False]

        def fail_after_removing_stage_contents(path, *arguments, **keywords):
            directory = Path(path)
            if directory.name.startswith(
                (".supercell-gate.", ".supercell-gate-stage.")
            ) and not stage_cleanup_failed[0]:
                stage_cleanup_failed[0] = True
                owner_directories = list(
                    self.root.glob(".supercell-gate-owner.*")
                )
                self.assertEqual(len(owner_directories), 1)
                owners = sorted(owner_directories[0].iterdir())
                self.assertEqual(len(owners), 2)
                for owner, final in zip(
                    owners, (self.h_dense_out, self.h_occ_out)
                ):
                    owner_stat = owner.stat()
                    final_stat = final.stat()
                    self.assertEqual(
                        (owner_stat.st_dev, owner_stat.st_ino),
                        (final_stat.st_dev, final_stat.st_ino),
                    )
                for child in directory.iterdir():
                    child.unlink()
                raise OSError("simulated partial stage cleanup failure")
            return real_rmtree(path, *arguments, **keywords)

        with mock.patch.object(
            shutil,
            "rmtree",
            side_effect=fail_after_removing_stage_contents,
        ):
            with self.assertRaisesRegex(
                OSError, "simulated partial stage cleanup failure"
            ):
                cli._publish_snapshots_together(
                    self.h_dense_out,
                    snapshot,
                    self.h_occ_out,
                    snapshot,
                )

        self.assertTrue(stage_cleanup_failed[0])
        self.assert_no_outputs()
        self.assertEqual(list(self.root.glob(".supercell-gate*")), [])

    def test_joint_publication_owner_cleanup_failure_is_postcommit(self):
        key = BlockKey(0, 0, (0, 0, 0))
        snapshot = _snapshot({key: np.ones((1, 1), dtype=np.float64)}, scalar="real64")
        real_rmtree = shutil.rmtree
        owner_cleanup_attempted = [False]

        def fail_owner_cleanup(path, *arguments, **keywords):
            if Path(path).name.startswith(".supercell-gate-owner."):
                owner_cleanup_attempted[0] = True
                raise OSError("simulated owner cleanup failure")
            return real_rmtree(path, *arguments, **keywords)

        with mock.patch.object(
            shutil,
            "rmtree",
            side_effect=fail_owner_cleanup,
        ):
            cli._publish_snapshots_together(
                self.h_dense_out,
                snapshot,
                self.h_occ_out,
                snapshot,
            )

        self.assertTrue(owner_cleanup_attempted[0])
        self.assertEqual(len(list(self.root.glob(".supercell-gate-owner.*"))), 1)
        self.assertEqual(list(self.root.glob(".supercell-gate-stage.*")), [])
        for output in (self.h_dense_out, self.h_occ_out):
            published = read_snapshot(output)
            np.testing.assert_array_equal(published.blocks[key], np.ones((1, 1)))

    def test_joint_publication_normal_path_removes_hidden_directories(self):
        key = BlockKey(0, 0, (0, 0, 0))
        snapshot = _snapshot({key: np.ones((1, 1), dtype=np.float64)}, scalar="real64")

        cli._publish_snapshots_together(
            self.h_dense_out,
            snapshot,
            self.h_occ_out,
            snapshot,
        )

        self.assertEqual(list(self.root.glob(".supercell-gate*")), [])

    def test_outputs_in_different_parents_are_rejected(self):
        self.write_valid_case()
        other_parent = self.root / "other"
        other_parent.mkdir()
        self.h_occ_out = other_parent / self.h_occ_out.name

        result = self.run_gate()

        self.assertEqual(result.returncode, 2)
        self.assertIn("same parent", result.stderr)
        self.assert_no_outputs()

    def test_sparse_reference_uses_zero_for_missing_blocks(self):
        key, expected_h, density_post = self.write_valid_case()
        self.period = (2, 1, 1)
        write_snapshot(self.h_reference_path, _snapshot({key: expected_h}, scalar="real64"))
        energy = self._explicit_dotc(density_post, expected_h)
        self.energy_reference_path.write_text(
            "{:.17e} {:.17e}\n".format(float(energy.real), float(energy.imag)),
            encoding="utf-8",
        )

        result = self.run_gate()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertLess(report["dense_H_rel_fro"], 1.0e-12)
        self.assertLess(report["occupied_H_rel_fro"], 1.0e-12)
        expected_keys = {
            BlockKey(0, 0, (0, 0, 0)),
            BlockKey(0, 0, (-1, 0, 0)),
        }
        self.assertEqual(set(read_snapshot(self.h_dense_out).blocks), expected_keys)
        np.testing.assert_array_equal(
            read_snapshot(self.h_dense_out).blocks[BlockKey(0, 0, (-1, 0, 0))],
            np.zeros_like(expected_h),
        )


if __name__ == "__main__":
    unittest.main()
