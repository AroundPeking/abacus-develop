"""Command-line gates for EXX snapshot comparison and occupied projection."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bvk import from_k, k_sector, to_k
from .io import BlockKey, Snapshot, TensorMap, read_snapshot, write_snapshot
from .occupied import OccupiedFactor, occupied_factor
from .supercell import (
    assemble_pair_coefficient,
    assemble_translation_matrix,
    direct_exchange,
    extract_reference_cell_blocks,
    infer_supercell_layout,
    map_dotc,
    occupied_exchange,
)


Period = Tuple[int, int, int]


def _require_finite_blocks(blocks: Mapping[object, np.ndarray], name: str) -> None:
    for block in blocks.values():
        if not np.isfinite(block).all():
            raise ValueError("{} tensor values must all be finite".format(name))


def read_real_scalar(path: Path) -> float:
    """Read a finite ``real imag`` scalar and enforce the replay reality gate."""

    fields = Path(path).read_text(encoding="utf-8").split()
    if len(fields) != 2:
        raise ValueError("energy scalar must contain exactly two numbers: real imag")
    try:
        real, imag = (float(value) for value in fields)
    except ValueError as error:
        raise ValueError("energy scalar contains a non-numeric value") from error
    if not math.isfinite(real) or not math.isfinite(imag):
        raise ValueError("energy scalar must contain only finite values")
    if abs(imag) >= 1.0e-13:
        raise ValueError("energy scalar imaginary part must have magnitude below 1e-13")
    return real


def _serial_numeric_snapshot(path: Path, name: str) -> Snapshot:
    snapshot = read_snapshot(path)
    if (snapshot.rank, snapshot.nranks) != (0, 1):
        raise ValueError("{} snapshot must be rank 0 of 1".format(name))
    if snapshot.scalar not in ("real64", "complex128"):
        raise ValueError("{} snapshot must use real64 or complex128 scalars".format(name))
    _require_finite_blocks(snapshot.blocks, "{} snapshot".format(name))
    return snapshot


def _infer_nat(blocks: Mapping[BlockKey, np.ndarray]) -> int:
    atoms = sorted({atom for key in blocks for atom in (key.ia1, key.ia2)})
    if not atoms or atoms != list(range(len(atoms))):
        raise ValueError("cannot infer nat from an empty or noncontiguous atom-index set")
    return len(atoms)


def _relative_frobenius(
    reference: Mapping[BlockKey, np.ndarray], candidate: Mapping[BlockKey, np.ndarray]
) -> Optional[float]:
    reference_squared = 0.0
    difference_squared = 0.0
    for key, reference_block in reference.items():
        candidate_block = candidate[key]
        reference_squared += float(np.vdot(reference_block, reference_block).real)
        difference = candidate_block - reference_block
        difference_squared += float(np.vdot(difference, difference).real)
    if reference_squared == 0.0:
        return 0.0 if difference_squared == 0.0 else None
    return math.sqrt(difference_squared / reference_squared)


def compare_snapshots(arguments: argparse.Namespace) -> dict:
    reference = _serial_numeric_snapshot(arguments.reference, "reference H")
    candidate = _serial_numeric_snapshot(arguments.candidate, "candidate H")
    if reference.scalar != candidate.scalar:
        raise ValueError("reference and candidate H snapshots have different scalar types")
    if set(reference.blocks) != set(candidate.blocks):
        raise ValueError("reference and candidate H snapshots have different key sets")
    for key, reference_block in reference.blocks.items():
        if reference_block.shape != candidate.blocks[key].shape:
            raise ValueError("reference and candidate H snapshots have different shape sets")

    nat = arguments.nat if arguments.nat is not None else _infer_nat(reference.blocks)
    if nat <= 0:
        raise ValueError("nat must be positive")
    h_relative = _relative_frobenius(reference.blocks, candidate.blocks)
    reference_energy = read_real_scalar(arguments.energy_reference)
    candidate_energy = read_real_scalar(arguments.energy_candidate)
    energy_absolute_per_atom = abs(candidate_energy - reference_energy) / nat
    passed = (
        h_relative is not None
        and h_relative <= arguments.h_rel_tol
        and energy_absolute_per_atom <= arguments.energy_abs_tol
    )
    return {
        "H_rel_fro": h_relative,
        "E_abs_Ry_atom": energy_absolute_per_atom,
        "nat": nat,
        "pass": passed,
    }


def _validate_tolerance(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and nonnegative".format(name))
    return result


def _fourier_roundtrip_max(blocks: Mapping[Tuple[int, int], np.ndarray], period: Period) -> float:
    """Check one atom pair at a time so no second full snapshot is retained."""

    maximum = 0.0
    for pair, original in blocks.items():
        with np.errstate(over="ignore", invalid="ignore"):
            roundtrip = to_k(from_k({pair: original}, period), period)
        if set(roundtrip) != {pair} or original.shape != roundtrip[pair].shape:
            return math.inf
        _require_finite_blocks(roundtrip, "Fourier roundtrip")
        if original.size:
            with np.errstate(over="ignore", invalid="ignore"):
                difference = np.abs(original - roundtrip[pair])
            if not np.isfinite(difference).all():
                raise ValueError("Fourier roundtrip difference must be finite")
            maximum = max(maximum, float(np.max(difference)))
    return maximum


def _density_dimensions(density_k: Mapping[Tuple[int, int], np.ndarray], period: Period):
    dimensions: Dict[int, int] = {}
    for (ia1, ia2), array in density_k.items():
        if array.ndim != 5 or tuple(array.shape[:3]) != period:
            raise ValueError("D blocks must be two-dimensional after the three period axes")
        rows, columns = array.shape[3:]
        for atom, dimension in ((ia1, rows), (ia2, columns)):
            if atom in dimensions and dimensions[atom] != dimension:
                raise ValueError("inconsistent D AO dimension for atom {}".format(atom))
            dimensions[atom] = dimension
    atoms = sorted(dimensions)
    if not atoms or atoms != list(range(len(atoms))):
        raise ValueError("D atom indices must be nonempty, contiguous, and zero based")
    return atoms, dimensions


def _coefficient_layout(
    coefficient_k: Mapping[Tuple[int, int], np.ndarray],
    period: Period,
    atoms: Sequence[int],
    ao_dimensions: Mapping[int, int],
):
    auxiliary_dimensions: Dict[int, int] = {}
    for (ia1, ia2), array in coefficient_k.items():
        if array.ndim != 6 or tuple(array.shape[:3]) != period:
            raise ValueError("C blocks must be three-dimensional after the three period axes")
        if ia1 not in ao_dimensions or ia2 not in ao_dimensions:
            raise ValueError("C contains an atom absent from D")
        naux, first_ao, second_ao = array.shape[3:]
        if first_ao != ao_dimensions[ia1] or second_ao != ao_dimensions[ia2]:
            raise ValueError("C block AO dimensions are incompatible with D")
        if ia1 in auxiliary_dimensions and auxiliary_dimensions[ia1] != naux:
            raise ValueError("inconsistent C auxiliary dimension for atom {}".format(ia1))
        auxiliary_dimensions[ia1] = naux
    if set(auxiliary_dimensions) != set(atoms):
        raise ValueError("C must provide auxiliary rows for every D atom")

    ao_offsets: Dict[int, int] = {}
    row_offsets: Dict[int, int] = {}
    ao_offset = 0
    row_offset = 0
    for atom in atoms:
        ao_offsets[atom] = ao_offset
        row_offsets[atom] = row_offset
        ao_offset += ao_dimensions[atom]
        row_offset += auxiliary_dimensions[atom] * ao_dimensions[atom]
    return auxiliary_dimensions, ao_offsets, row_offsets, ao_offset, row_offset


def _assemble_density(
    sector: Mapping[Tuple[int, int], np.ndarray], atoms: Sequence[int], dimensions, offsets, total
) -> np.ndarray:
    matrix = np.zeros((total, total), dtype=np.complex128)
    for (ia1, ia2), block in sector.items():
        row = offsets[ia1]
        column = offsets[ia2]
        matrix[row : row + dimensions[ia1], column : column + dimensions[ia2]] = block
    return matrix


def _assemble_coefficient(
    sector: Mapping[Tuple[int, int], np.ndarray],
    ao_dimensions,
    auxiliary_dimensions,
    ao_offsets,
    row_offsets,
    total_ao,
    total_rows,
) -> np.ndarray:
    matrix = np.zeros((total_rows, total_ao), dtype=np.complex128)
    for (ia1, ia2), block in sector.items():
        row = row_offsets[ia1]
        column = ao_offsets[ia2]
        rows = auxiliary_dimensions[ia1] * ao_dimensions[ia1]
        matrix[row : row + rows, column : column + ao_dimensions[ia2]] = block.reshape(
            rows, ao_dimensions[ia2]
        )
    return matrix


def _disassemble_coefficient(
    matrix: np.ndarray,
    pairs,
    ao_dimensions,
    auxiliary_dimensions,
    ao_offsets,
    row_offsets,
):
    sector = {}
    for ia1, ia2 in pairs:
        rows = auxiliary_dimensions[ia1] * ao_dimensions[ia1]
        row = row_offsets[ia1]
        column = ao_offsets[ia2]
        sector[(ia1, ia2)] = matrix[
            row : row + rows, column : column + ao_dimensions[ia2]
        ].reshape(auxiliary_dimensions[ia1], ao_dimensions[ia1], ao_dimensions[ia2])
    return sector


def _snapshot_byte_estimate(blocks: Mapping[BlockKey, np.ndarray]) -> int:
    return 32 + sum(24 + 8 * block.ndim + 8 + block.nbytes for block in blocks.values())


def _real_projected_blocks(blocks: TensorMap) -> TensorMap:
    result = {}
    for key, block in blocks.items():
        real_scale = max(float(np.max(np.abs(block.real))) if block.size else 0.0, 1.0)
        imaginary_maximum = float(np.max(np.abs(block.imag))) if block.size else 0.0
        if imaginary_maximum > 1.0e-12 * real_scale:
            raise ValueError("real64 projected C has a non-negligible imaginary component")
        result[key] = np.asarray(block.real, dtype=np.float64, order="C")
    return result


def project_snapshot(arguments: argparse.Namespace) -> dict:
    period = tuple(int(value) for value in arguments.period)
    if any(value <= 0 for value in period):
        raise ValueError("period must contain three positive integers")
    eigenvalue_tol = _validate_tolerance(arguments.eigenvalue_tol, "eigenvalue tolerance")
    coefficient = _serial_numeric_snapshot(arguments.C, "C")
    density = _serial_numeric_snapshot(arguments.D_full, "D.full")
    if coefficient.scalar != density.scalar:
        raise ValueError("C and D.full snapshots must use the same scalar type")
    output_scalar = coefficient.scalar

    with np.errstate(over="ignore", invalid="ignore"):
        coefficient_k = to_k(coefficient.blocks, period)
    del coefficient
    _require_finite_blocks(coefficient_k, "Fourier-transformed C")
    with np.errstate(over="ignore", invalid="ignore"):
        density_k = to_k(density.blocks, period)
    del density
    _require_finite_blocks(density_k, "Fourier-transformed D.full")
    c_fourier_roundtrip_max = _fourier_roundtrip_max(coefficient_k, period)
    d_fourier_roundtrip_max = _fourier_roundtrip_max(density_k, period)
    fourier_roundtrip_max = max(c_fourier_roundtrip_max, d_fourier_roundtrip_max)
    atoms, ao_dimensions = _density_dimensions(density_k, period)
    (
        auxiliary_dimensions,
        ao_offsets,
        row_offsets,
        total_ao,
        total_rows,
    ) = _coefficient_layout(coefficient_k, period, atoms, ao_dimensions)

    hermiticity_max = 0.0
    minimum_eigenvalue = math.inf
    scaled_minima = []
    scaled_minimum_undefined = False
    psd_pass = True
    ranks = []
    discarded_trace = 0.0
    factors = []
    densities = []
    for k_index in np.ndindex(period):
        density_matrix = _assemble_density(
            k_sector(density_k, k_index), atoms, ao_dimensions, ao_offsets, total_ao
        )
        with np.errstate(over="ignore", invalid="ignore"):
            antihermitian = density_matrix - density_matrix.conj().T
        if not np.isfinite(antihermitian).all():
            raise ValueError("D Hermiticity residual must be finite")
        hermiticity = float(np.max(np.abs(antihermitian)))
        hermiticity_max = max(hermiticity_max, hermiticity)
        with np.errstate(over="ignore", invalid="ignore"):
            hermitian_density = 0.5 * density_matrix + 0.5 * density_matrix.conj().T
        if not np.isfinite(hermitian_density).all():
            raise ValueError("Hermitian D matrix must be finite")
        eigenvalues = np.linalg.eigvalsh(hermitian_density)
        positive_maximum = max(float(np.max(eigenvalues)), 0.0)
        minimum = float(np.min(eigenvalues))
        minimum_eigenvalue = min(minimum_eigenvalue, minimum)
        if positive_maximum > 0.0:
            scaled_minima.append(minimum / positive_maximum)
            psd_pass = psd_pass and minimum >= -1.0e-10 * positive_maximum
        elif minimum < 0.0:
            scaled_minimum_undefined = True
            psd_pass = False
        else:
            scaled_minima.append(0.0)
        densities.append(density_matrix)

    minimum_scaled_eigenvalue = (
        None if scaled_minimum_undefined else min(scaled_minima)
    )

    gates_pass = (
        c_fourier_roundtrip_max < 1.0e-13
        and d_fourier_roundtrip_max < 1.0e-13
        and hermiticity_max < 1.0e-12
        and psd_pass
    )
    projected_blocks: Optional[TensorMap] = None
    if gates_pass:
        for density_matrix in densities:
            factor = occupied_factor(density_matrix, eigenvalue_tol)
            factors.append(factor)
            ranks.append(int(factor.O.shape[1]))
            discarded_trace += factor.discarded_trace

        pairs = [
            (row_atom, column_atom)
            for row_atom in sorted(auxiliary_dimensions)
            for column_atom in atoms
        ]
        projected_k = {
            (ia1, ia2): np.zeros(
                period + (auxiliary_dimensions[ia1], ao_dimensions[ia1], ao_dimensions[ia2]),
                dtype=np.complex128,
            )
            for ia1, ia2 in pairs
        }
        for linear_index, k_index in enumerate(np.ndindex(period)):
            coefficient_matrix = _assemble_coefficient(
                k_sector(coefficient_k, k_index),
                ao_dimensions,
                auxiliary_dimensions,
                ao_offsets,
                row_offsets,
                total_ao,
                total_rows,
            )
            factor: OccupiedFactor = factors[linear_index]
            with np.errstate(over="ignore", invalid="ignore"):
                projected_matrix = coefficient_matrix @ (factor.O @ factor.O_pinv)
            if not np.isfinite(projected_matrix).all():
                raise ValueError("projected C matrix must be finite")
            projected_sector = _disassemble_coefficient(
                projected_matrix,
                pairs,
                ao_dimensions,
                auxiliary_dimensions,
                ao_offsets,
                row_offsets,
            )
            for pair, block in projected_sector.items():
                projected_k[pair][k_index] = block
        with np.errstate(over="ignore", invalid="ignore"):
            projected_blocks = from_k(projected_k, period)
        _require_finite_blocks(projected_blocks, "projected C snapshot")
        if output_scalar == "real64":
            projected_blocks = _real_projected_blocks(projected_blocks)
        write_snapshot(arguments.output, Snapshot(1, output_scalar, 0, 1, projected_blocks))

    return {
        "fourier_roundtrip_max": fourier_roundtrip_max,
        "C_fourier_roundtrip_max": c_fourier_roundtrip_max,
        "D_fourier_roundtrip_max": d_fourier_roundtrip_max,
        "D_hermiticity_max": hermiticity_max,
        "D_min_eigenvalue": minimum_eigenvalue,
        "D_min_eigenvalue_scaled": minimum_scaled_eigenvalue,
        "occupied_ranks_by_k": ranks,
        "discarded_trace": discarded_trace,
        "output_blocks": 0 if projected_blocks is None else len(projected_blocks),
        "output_bytes_estimate": 0 if projected_blocks is None else _snapshot_byte_estimate(projected_blocks),
        "pass": gates_pass,
    }


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _normalized_sumsq(values: np.ndarray) -> Tuple[float, float]:
    """Represent one real-array Frobenius norm as ``scale * sqrt(sumsq)``."""

    if not np.isfinite(values).all():
        raise ValueError("Frobenius norm input must contain only finite values")
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale == 0.0:
        return 0.0, 0.0
    normalized = values / scale
    sumsq = float(np.sum(normalized * normalized, dtype=np.float64))
    if not math.isfinite(sumsq):
        raise ValueError("Frobenius norm must be finite")
    return scale, sumsq


def _combine_sumsq(
    total: Tuple[float, float], contribution: Tuple[float, float]
) -> Tuple[float, float]:
    total_scale, total_sumsq = total
    scale, sumsq = contribution
    if scale == 0.0 or sumsq == 0.0:
        return total
    if total_scale == 0.0 or total_sumsq == 0.0:
        return scale, sumsq
    if total_scale < scale:
        ratio = total_scale / scale
        return scale, sumsq + total_sumsq * ratio * ratio
    ratio = scale / total_scale
    return total_scale, total_sumsq + sumsq * ratio * ratio


def _complex_norm_pair(array: np.ndarray) -> Tuple[float, float]:
    total = (0.0, 0.0)
    total = _combine_sumsq(total, _normalized_sumsq(array.real))
    total = _combine_sumsq(total, _normalized_sumsq(array.imag))
    return total


def _complex_difference_norm_pair(
    reference: np.ndarray, candidate: np.ndarray
) -> Tuple[float, float]:
    total = (0.0, 0.0)
    for reference_part, candidate_part in (
        (reference.real, candidate.real),
        (reference.imag, candidate.imag),
    ):
        if not np.isfinite(reference_part).all() or not np.isfinite(candidate_part).all():
            raise ValueError("Frobenius difference input must contain only finite values")
        reference_scale = float(np.max(np.abs(reference_part))) if reference_part.size else 0.0
        candidate_scale = float(np.max(np.abs(candidate_part))) if candidate_part.size else 0.0
        scale = max(reference_scale, candidate_scale)
        if scale == 0.0:
            continue
        normalized_difference = candidate_part / scale - reference_part / scale
        sumsq = float(
            np.sum(normalized_difference * normalized_difference, dtype=np.float64)
        )
        if not math.isfinite(sumsq):
            raise ValueError("Frobenius difference norm must be finite")
        total = _combine_sumsq(total, (scale, sumsq))
    return total


def _norm_ratio(
    numerator: Tuple[float, float], denominator: Tuple[float, float]
) -> Optional[float]:
    numerator_scale, numerator_sumsq = numerator
    denominator_scale, denominator_sumsq = denominator
    if denominator_scale == 0.0 or denominator_sumsq == 0.0:
        return 0.0 if numerator_scale == 0.0 or numerator_sumsq == 0.0 else None
    if numerator_scale == 0.0 or numerator_sumsq == 0.0:
        return 0.0
    ratio = (numerator_scale / denominator_scale) * math.sqrt(
        numerator_sumsq / denominator_sumsq
    )
    if not math.isfinite(ratio):
        raise ValueError("relative Frobenius norm must be finite")
    return ratio


def _relative_frobenius_arrays(reference: np.ndarray, candidate: np.ndarray) -> Optional[float]:
    if reference.shape != candidate.shape:
        raise ValueError("Frobenius comparison arrays must have identical shapes")
    return _norm_ratio(
        _complex_difference_norm_pair(reference, candidate),
        _complex_norm_pair(reference),
    )


def _relative_frobenius_union(
    reference: Mapping[BlockKey, np.ndarray], candidate: Mapping[BlockKey, np.ndarray]
) -> Optional[float]:
    reference_norm = (0.0, 0.0)
    difference_norm = (0.0, 0.0)
    keys = sorted(
        set(reference).union(candidate),
        key=lambda key: (key.ia1, key.ia2, key.R),
    )
    for key in keys:
        reference_block = reference.get(key)
        candidate_block = candidate.get(key)
        if reference_block is not None and candidate_block is not None:
            if reference_block.shape != candidate_block.shape:
                raise ValueError("H snapshot union contains incompatible matching shapes")
        elif reference_block is None:
            reference_block = np.zeros_like(candidate_block)
        else:
            candidate_block = np.zeros_like(reference_block)
        reference_norm = _combine_sumsq(
            reference_norm, _complex_norm_pair(reference_block)
        )
        difference_norm = _combine_sumsq(
            difference_norm,
            _complex_difference_norm_pair(reference_block, candidate_block),
        )
    return _norm_ratio(difference_norm, reference_norm)


def _validate_output_paths(dense_path: Path, occupied_path: Path) -> Path:
    dense = Path(dense_path)
    occupied = Path(occupied_path)
    dense_parent = dense.parent.resolve()
    occupied_parent = occupied.parent.resolve()
    if dense_parent != occupied_parent:
        raise ValueError("H outputs must have the same parent directory")
    if dense.resolve() == occupied.resolve():
        raise ValueError("H outputs must be distinct paths")
    for output in (dense, occupied):
        if os.path.lexists(str(output)):
            raise ValueError("output already exists: {}".format(output))
    return dense_parent


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_inode_as_descriptor(descriptor: int, path: Path) -> bool:
    try:
        owner_stat = os.fstat(descriptor)
        path_stat = os.stat(str(path), follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (owner_stat.st_dev, owner_stat.st_ino) == (
        path_stat.st_dev,
        path_stat.st_ino,
    )


def _close_descriptors(descriptors: Sequence[int]) -> None:
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _publish_snapshots_together(
    dense_path: Path,
    dense_snapshot: Snapshot,
    occupied_path: Path,
    occupied_snapshot: Snapshot,
) -> None:
    """Publish two snapshots exclusively, rolling back only our own hard links."""

    dense = Path(dense_path)
    occupied = Path(occupied_path)
    parent = _validate_output_paths(dense, occupied)
    stage_directory = Path(
        tempfile.mkdtemp(prefix=".supercell-gate.", dir=str(parent))
    )
    descriptors = []
    owned_publications = []
    try:
        dense_stage = stage_directory / "H.dense.stage.exxcmp"
        occupied_stage = stage_directory / "H.occ.stage.exxcmp"
        write_snapshot(dense_stage, dense_snapshot)
        write_snapshot(occupied_stage, occupied_snapshot)
        stages = ((dense_stage, dense), (occupied_stage, occupied))
        for stage, final in stages:
            descriptor = os.open(str(stage), os.O_RDONLY)
            descriptors.append(descriptor)
            owned_publications.append((descriptor, final))
        for stage, final in stages:
            os.link(str(stage), str(final))
        _fsync_directory(parent)
        shutil.rmtree(stage_directory)
    except BaseException:
        for descriptor, final in owned_publications:
            try:
                if _same_inode_as_descriptor(descriptor, final):
                    final.unlink()
            except OSError:
                pass
        try:
            _fsync_directory(parent)
        except OSError:
            pass
        try:
            shutil.rmtree(stage_directory)
        except OSError:
            pass
        _close_descriptors(descriptors)
        raise
    _close_descriptors(descriptors)


def _supercell_report(
    max_elements: int, live_elements: int, snapshot_elements: int, nat: int
) -> dict:
    return {
        "dense_H_rel_fro": None,
        "occupied_H_rel_fro": None,
        "dense_occ_H_rel_fro": None,
        "dense_E_abs_Ry_atom": None,
        "occupied_E_abs_Ry_atom": None,
        "occupied_ranks_by_supercell": [],
        "discarded_trace": None,
        "D_hermiticity_rel_fro": None,
        "D_min_eigenvalue": None,
        "D_min_eigenvalue_scaled": None,
        "density_factor_rel_fro": None,
        "dense_bytes": 0,
        "occupied_bytes": 0,
        "live_elements_upper_bound": live_elements,
        "snapshot_elements_upper_bound": snapshot_elements,
        "max_elements": max_elements,
        "nat": nat,
        "pass": False,
    }


def _real_map_energy(density_blocks: TensorMap, h_blocks: TensorMap) -> float:
    energy = map_dotc(density_blocks, h_blocks)
    if abs(energy.imag) >= 1.0e-13:
        raise ValueError("exchange energy imaginary part must have magnitude below 1e-13")
    if not math.isfinite(energy.real):
        raise ValueError("exchange energy must be finite")
    return float(energy.real)


def _absolute_energy_error_per_atom(candidate: float, reference: float, nat: int) -> float:
    difference = abs(candidate - reference) / nat
    if not math.isfinite(difference):
        raise ValueError("exchange energy error per atom must be finite")
    return difference


def _snapshot_elements_upper_bound(paths: Sequence[Path]) -> int:
    return sum((Path(path).stat().st_size + 15) // 16 for path in paths)


def _validate_ao_blocks_against_layout(
    blocks: Mapping[BlockKey, np.ndarray], layout, name: str
) -> None:
    layout_atoms = set(layout.atoms)
    block_atoms = set()
    for key, block in blocks.items():
        if key.ia1 not in layout_atoms or key.ia2 not in layout_atoms:
            raise ValueError(
                "{} atom universe must match the supercell layout".format(name)
            )
        expected_shape = (
            layout.ao_dimensions[key.ia1],
            layout.ao_dimensions[key.ia2],
        )
        if block.shape != expected_shape:
            raise ValueError(
                "{} block shape is incompatible with the supercell layout".format(name)
            )
        block_atoms.update((key.ia1, key.ia2))
    if block_atoms != layout_atoms:
        raise ValueError(
            "{} atom universe must match the supercell layout".format(name)
        )


def supercell_gate(arguments: argparse.Namespace) -> dict:
    dense_output = Path(arguments.H_dense_out)
    occupied_output = Path(arguments.H_occ_out)
    _validate_output_paths(dense_output, occupied_output)

    period = tuple(int(value) for value in arguments.period)
    if any(value <= 0 for value in period):
        raise ValueError("period must contain three positive integers")
    max_elements = int(arguments.max_elements)
    nk = math.prod(period)
    if nk * nk > max_elements:
        raise ValueError("supercell cell-pair count exceeds max_elements")

    snapshot_paths = (
        arguments.C,
        arguments.V,
        arguments.D_full,
        arguments.D_post,
        arguments.H_reference,
    )
    snapshot_elements = _snapshot_elements_upper_bound(snapshot_paths)
    if snapshot_elements > max_elements:
        raise ValueError("dense supercell live allocation exceeds max_elements")

    coefficient = _serial_numeric_snapshot(arguments.C, "C")
    metric = _serial_numeric_snapshot(arguments.V, "V")
    density = _serial_numeric_snapshot(arguments.D_full, "D.full")
    density_post = _serial_numeric_snapshot(arguments.D_post, "D.post")
    h_reference = _serial_numeric_snapshot(arguments.H_reference, "reference H")
    snapshots = (coefficient, metric, density, density_post, h_reference)
    scalar = coefficient.scalar
    if any(snapshot.scalar != scalar for snapshot in snapshots[1:]):
        raise ValueError("all supercell-gate snapshots must use the same scalar type")
    energy_reference = read_real_scalar(arguments.energy_reference)

    layout = infer_supercell_layout(
        coefficient.blocks, metric.blocks, density.blocks, period
    )
    _validate_ao_blocks_against_layout(
        h_reference.blocks, layout, "reference H"
    )
    _validate_ao_blocks_against_layout(
        density_post.blocks, layout, "D.post"
    )
    nat = len(layout.atoms)
    nP = int(layout.naux_supercell)
    nA = int(layout.nao_supercell)
    coefficient_elements = nP * nA * nA
    metric_elements = nP * nP
    density_elements = nA * nA
    factor_peak = coefficient_elements + metric_elements + 10 * density_elements
    if scalar == "real64":
        direct_peak = (
            4 * coefficient_elements + 2 * metric_elements + 4 * density_elements
        )
        occupied_peak = (
            4 * coefficient_elements + 2 * metric_elements + 4 * density_elements
        )
    else:
        direct_peak = (
            3 * coefficient_elements + metric_elements + 3 * density_elements
        )
        occupied_peak = (
            3 * coefficient_elements + metric_elements + 4 * density_elements
        )
    # This is a complex128-equivalent element bound, not total Python RSS.  It
    # includes the five snapshot file payloads; factor_peak also reserves
    # 4 * D elements for eigensolver/workspace needs.
    core_elements = max(factor_peak, direct_peak, occupied_peak)
    live_elements = snapshot_elements + core_elements
    if live_elements > max_elements:
        raise ValueError("dense supercell live allocation exceeds max_elements")
    report = _supercell_report(
        max_elements, live_elements, snapshot_elements, nat
    )

    dense_coefficient = assemble_pair_coefficient(
        coefficient.blocks, layout, max_elements
    )
    dense_metric = assemble_translation_matrix(
        metric.blocks, layout, "auxiliary", max_elements
    )
    dense_density = assemble_translation_matrix(
        density.blocks, layout, "ao", max_elements
    )
    del coefficient, metric, density

    hermiticity = _relative_frobenius_arrays(
        dense_density, dense_density.conj().T
    )
    if hermiticity is None:
        raise ValueError("D Hermiticity residual is undefined")
    report["D_hermiticity_rel_fro"] = hermiticity
    if hermiticity > 1.0e-12:
        return report

    with np.errstate(over="ignore", invalid="ignore"):
        hermitian_density = 0.5 * dense_density + 0.5 * dense_density.conj().T
    del dense_density
    if not np.isfinite(hermitian_density).all():
        raise ValueError("Hermitian D matrix must contain only finite values")
    try:
        eigenvalues = np.linalg.eigvalsh(hermitian_density)
    except np.linalg.LinAlgError as error:
        raise ValueError("Hermitian D eigensolver failed") from error
    if not np.isfinite(eigenvalues).all():
        raise ValueError("D eigenvalues must contain only finite values")
    minimum = float(np.min(eigenvalues))
    maximum = float(np.max(eigenvalues))
    report["D_min_eigenvalue"] = minimum
    if maximum > 0.0:
        scaled_minimum = minimum / maximum
        if not math.isfinite(scaled_minimum):
            scaled_minimum = math.copysign(sys.float_info.max, minimum)
        report["D_min_eigenvalue_scaled"] = scaled_minimum
        psd_pass = minimum >= -1.0e-10 * maximum
    elif minimum < 0.0:
        report["D_min_eigenvalue_scaled"] = None
        psd_pass = False
    else:
        report["D_min_eigenvalue_scaled"] = 0.0
        psd_pass = True
    del eigenvalues
    if not psd_pass:
        return report

    factor = occupied_factor(hermitian_density, 0.0)
    occupied_orbitals = factor.O
    report["occupied_ranks_by_supercell"] = [int(occupied_orbitals.shape[1])]
    report["discarded_trace"] = float(factor.discarded_trace)
    del factor
    with np.errstate(over="ignore", invalid="ignore"):
        factored_density = occupied_orbitals @ occupied_orbitals.conj().T
    if not np.isfinite(factored_density).all():
        raise ValueError("occupied density factor product must contain only finite values")
    density_factor_relative = _relative_frobenius_arrays(
        hermitian_density, factored_density
    )
    del factored_density
    if density_factor_relative is None:
        raise ValueError("density factor relative Frobenius residual is undefined")
    report["density_factor_rel_fro"] = density_factor_relative
    if density_factor_relative > 1.0e-12:
        return report

    direct = direct_exchange(dense_coefficient, dense_metric, hermitian_density)
    report["dense_bytes"] = 16 * int(direct.temporary_elements)
    dense_matrix = direct.matrix
    del direct

    occupied = occupied_exchange(
        dense_coefficient, dense_metric, occupied_orbitals
    )
    report["occupied_bytes"] = 16 * int(occupied.temporary_elements)
    occupied_matrix = occupied.matrix
    del occupied, occupied_orbitals, dense_coefficient, dense_metric, hermitian_density

    dense_blocks = extract_reference_cell_blocks(dense_matrix, layout, scalar)
    occupied_blocks = extract_reference_cell_blocks(occupied_matrix, layout, scalar)
    del dense_matrix, occupied_matrix
    dense_h_relative = _relative_frobenius_union(h_reference.blocks, dense_blocks)
    occupied_h_relative = _relative_frobenius_union(
        h_reference.blocks, occupied_blocks
    )
    dense_occupied_relative = _relative_frobenius_union(
        dense_blocks, occupied_blocks
    )
    report["dense_H_rel_fro"] = dense_h_relative
    report["occupied_H_rel_fro"] = occupied_h_relative
    report["dense_occ_H_rel_fro"] = dense_occupied_relative

    dense_energy = _real_map_energy(density_post.blocks, dense_blocks)
    occupied_energy = _real_map_energy(density_post.blocks, occupied_blocks)
    report["dense_E_abs_Ry_atom"] = _absolute_energy_error_per_atom(
        dense_energy, energy_reference, nat
    )
    report["occupied_E_abs_Ry_atom"] = _absolute_energy_error_per_atom(
        occupied_energy, energy_reference, nat
    )
    passed = (
        dense_h_relative is not None
        and dense_h_relative <= 1.0e-10
        and occupied_h_relative is not None
        and occupied_h_relative <= 1.0e-10
        and dense_occupied_relative is not None
        and dense_occupied_relative <= 1.0e-12
        and report["dense_E_abs_Ry_atom"] <= 1.0e-10
        and report["occupied_E_abs_Ry_atom"] <= 1.0e-10
    )
    report["pass"] = passed
    if passed:
        _publish_snapshots_together(
            dense_output,
            Snapshot(1, scalar, 0, 1, dense_blocks),
            occupied_output,
            Snapshot(1, scalar, 0, 1, occupied_blocks),
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m exx_thc.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    compare = commands.add_parser("compare", help="compare replay H and energy outputs")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--energy-reference", type=Path, required=True)
    compare.add_argument("--energy-candidate", type=Path, required=True)
    compare.add_argument("--nat", type=int)
    compare.add_argument("--h-rel-tol", type=float, default=1.0e-12)
    compare.add_argument("--energy-abs-tol", type=float, default=1.0e-12)

    project = commands.add_parser("project", help="project C onto the occupied D space")
    project.add_argument("--C", type=Path, required=True)
    project.add_argument("--D-full", dest="D_full", type=Path, required=True)
    project.add_argument("--period", type=int, nargs=3, metavar=("PX", "PY", "PZ"), required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--eigenvalue-tol", type=float, default=0.0)

    supercell = commands.add_parser(
        "supercell-gate", help="gate dense and occupied finite-supercell EXX"
    )
    supercell.add_argument("--C", type=Path, required=True)
    supercell.add_argument("--V", type=Path, required=True)
    supercell.add_argument("--D-full", dest="D_full", type=Path, required=True)
    supercell.add_argument("--D-post", dest="D_post", type=Path, required=True)
    supercell.add_argument(
        "--H-reference", dest="H_reference", type=Path, required=True
    )
    supercell.add_argument(
        "--energy-reference", dest="energy_reference", type=Path, required=True
    )
    supercell.add_argument(
        "--period", type=int, nargs=3, metavar=("PX", "PY", "PZ"), required=True
    )
    supercell.add_argument(
        "--H-dense-out", dest="H_dense_out", type=Path, required=True
    )
    supercell.add_argument(
        "--H-occ-out", dest="H_occ_out", type=Path, required=True
    )
    supercell.add_argument("--max-elements", type=_positive_integer, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "compare":
            arguments.h_rel_tol = _validate_tolerance(arguments.h_rel_tol, "H relative tolerance")
            arguments.energy_abs_tol = _validate_tolerance(
                arguments.energy_abs_tol, "energy absolute tolerance"
            )
            report = compare_snapshots(arguments)
        elif arguments.command == "project":
            report = project_snapshot(arguments)
        else:
            report = supercell_gate(arguments)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if report["pass"] else 1
    except (OSError, TypeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
