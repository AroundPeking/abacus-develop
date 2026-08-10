"""Born-von Karman transforms for atom-pair tensor blocks."""

from __future__ import annotations

import itertools
import operator
import sys
from typing import Dict, Mapping, Tuple

import numpy as np

from .io import BlockKey, TensorMap


Pair = Tuple[int, int]
if sys.version_info >= (3, 9):
    KTensorMap = dict[Pair, np.ndarray]
else:  # pragma: no cover - exercised only by the supported Python 3.8 runtime
    KTensorMap = Dict[Pair, np.ndarray]


def _validated_period(period: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(period, tuple) or len(period) != 3:
        raise ValueError("period must be a tuple of three positive integers")
    result = []
    for value in period:
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise ValueError("period must contain only positive integers") from error
        if integer <= 0:
            raise ValueError("period must contain only positive integers")
        result.append(int(integer))
    return result[0], result[1], result[2]


def _canonical_coordinate(value: int, period: int) -> int:
    residue = value % period
    return residue if residue <= (period - 1) // 2 else residue - period


def _canonical_cell(cell: tuple[int, int, int], period: tuple[int, int, int]):
    if not isinstance(cell, tuple) or len(cell) != 3:
        raise ValueError("R must be a tuple of three integers")
    result = []
    for value, extent in zip(cell, period):
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise ValueError("R must contain only integers") from error
        result.append(_canonical_coordinate(int(integer), extent))
    return result[0], result[1], result[2]


def _canonical_translations(period: tuple[int, int, int]):
    coordinates = []
    for extent in period:
        coordinates.append(
            tuple(_canonical_coordinate(index, extent) for index in range(extent))
        )
    return itertools.product(*coordinates)


def to_k(blocks: Mapping[BlockKey, np.ndarray], period: tuple[int, int, int]) -> KTensorMap:
    """Transform real-space blocks to arrays shaped ``period + block_shape``.

    Translation aliases modulo ``period`` are summed.  Missing translations are
    zeros, and the forward phase is ``exp(-2j*pi*dot(k/period, R))``.
    """

    period = _validated_period(period)
    canonical = {}
    shapes = {}
    for key, value in blocks.items():
        if not isinstance(key, BlockKey):
            raise TypeError("real-space block keys must be BlockKey instances")
        array = np.asarray(value)
        pair = (key.ia1, key.ia2)
        if pair in shapes and shapes[pair] != array.shape:
            raise ValueError("inconsistent block shapes for atom pair {}".format(pair))
        shapes[pair] = array.shape
        cell = _canonical_cell(key.R, period)
        aggregate_key = (pair, cell)
        converted = np.asarray(array, dtype=np.complex128)
        if aggregate_key in canonical:
            canonical[aggregate_key] += converted
        else:
            canonical[aggregate_key] = converted.copy()

    transformed: KTensorMap = {}
    period_array = np.asarray(period, dtype=np.float64)
    for pair, block_shape in shapes.items():
        values = np.zeros(period + block_shape, dtype=np.complex128)
        pair_blocks = [
            (cell, array)
            for (candidate_pair, cell), array in canonical.items()
            if candidate_pair == pair
        ]
        for k_index in np.ndindex(period):
            fractional_k = np.asarray(k_index, dtype=np.float64) / period_array
            sector = values[k_index]
            for cell, array in pair_blocks:
                phase = np.exp(-2j * np.pi * np.dot(fractional_k, cell))
                sector += phase * array
        transformed[pair] = values
    return transformed


def from_k(blocks: Mapping[Pair, np.ndarray], period: tuple[int, int, int]) -> TensorMap:
    """Inverse-transform KTensorMap arrays to one canonical block per BvK cell.

    Canonical coordinates are centered with the positive tie excluded; for
    example, period two uses representatives ``(0, -1)``.  The inverse carries
    ``1/prod(period)`` and the opposite Fourier phase.
    """

    period = _validated_period(period)
    nk = int(np.prod(period))
    period_array = np.asarray(period, dtype=np.float64)
    restored: TensorMap = {}
    for pair, value in blocks.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("KTensorMap keys must be two-atom tuples")
        array = np.asarray(value)
        if array.ndim < 3 or tuple(array.shape[:3]) != period:
            raise ValueError("KTensorMap array leading shape must equal period")
        block_shape = array.shape[3:]
        for cell in _canonical_translations(period):
            result = np.zeros(block_shape, dtype=np.complex128)
            for k_index in np.ndindex(period):
                fractional_k = np.asarray(k_index, dtype=np.float64) / period_array
                phase = np.exp(2j * np.pi * np.dot(fractional_k, cell))
                result += phase * np.asarray(array[k_index], dtype=np.complex128)
            restored[BlockKey(pair[0], pair[1], cell)] = result / nk
    return restored


def k_sector(blocks: Mapping[Pair, np.ndarray], k_index: tuple[int, int, int]):
    """Extract one atom-pair block mapping from a KTensorMap."""

    if not isinstance(k_index, tuple) or len(k_index) != 3:
        raise ValueError("k_index must be a tuple of three integers")
    sector = {}
    for pair, value in blocks.items():
        array = np.asarray(value)
        try:
            selected = array[k_index]
        except (IndexError, TypeError) as error:
            raise ValueError("k_index is outside the KTensorMap period") from error
        sector[pair] = np.asarray(selected, dtype=np.complex128)
    return sector


def assemble_matrix(blocks: Mapping[Pair, np.ndarray]) -> np.ndarray:
    """Assemble sorted 2D atom-pair blocks, filling absent pairs with zero."""

    atom_sizes = {}
    validated = {}
    for pair, value in blocks.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("atom-pair keys must contain exactly two atoms")
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError("assemble_matrix requires two-dimensional blocks")
        ia1, ia2 = pair
        for atom, size in ((ia1, array.shape[0]), (ia2, array.shape[1])):
            if atom in atom_sizes and atom_sizes[atom] != size:
                raise ValueError("inconsistent AO dimension for atom {}".format(atom))
            atom_sizes[atom] = size
        validated[pair] = np.asarray(array, dtype=np.complex128)

    atoms = sorted(atom_sizes)
    offsets = {}
    offset = 0
    for atom in atoms:
        offsets[atom] = offset
        offset += atom_sizes[atom]
    matrix = np.zeros((offset, offset), dtype=np.complex128)
    for (ia1, ia2), array in validated.items():
        row = offsets[ia1]
        column = offsets[ia2]
        matrix[
            row : row + atom_sizes[ia1], column : column + atom_sizes[ia2]
        ] = array
    return matrix
