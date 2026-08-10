"""Command-line gates for EXX snapshot comparison and occupied projection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bvk import from_k, k_sector, to_k
from .io import BlockKey, Snapshot, TensorMap, read_snapshot, write_snapshot
from .occupied import OccupiedFactor, occupied_factor


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
        else:
            report = project_snapshot(arguments)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if report["pass"] else 1
    except (OSError, TypeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
