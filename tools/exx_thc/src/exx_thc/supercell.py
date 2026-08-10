"""Finite-supercell layouts and dense EXX reference-map assembly."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Tuple

import numpy as np

from .io import BlockKey


Cell = Tuple[int, int, int]
OffsetKey = Tuple[Cell, int]
BlockRecord = Tuple[int, int, Cell, np.ndarray]


@dataclass(frozen=True)
class SupercellLayout:
    """Immutable cell ordering, per-atom dimensions, and dense offsets."""

    period: Cell
    cells: Tuple[Cell, ...]
    atoms: Tuple[int, ...]
    ao_dimensions: Mapping[int, int]
    auxiliary_dimensions: Mapping[int, int]
    ao_offsets: Mapping[OffsetKey, int]
    auxiliary_offsets: Mapping[OffsetKey, int]
    nao_supercell: int
    naux_supercell: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ao_dimensions", MappingProxyType(dict(self.ao_dimensions))
        )
        object.__setattr__(
            self,
            "auxiliary_dimensions",
            MappingProxyType(dict(self.auxiliary_dimensions)),
        )
        object.__setattr__(self, "ao_offsets", MappingProxyType(dict(self.ao_offsets)))
        object.__setattr__(
            self,
            "auxiliary_offsets",
            MappingProxyType(dict(self.auxiliary_offsets)),
        )


def _validated_period(period: object) -> Cell:
    if not isinstance(period, tuple) or len(period) != 3:
        raise ValueError("period must be a tuple of three positive integers")
    result = []
    for value in period:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("period must contain only positive integers")
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise TypeError("period must contain only positive integers") from error
        if integer <= 0:
            raise ValueError("period must contain only positive integers")
        result.append(int(integer))
    return result[0], result[1], result[2]


def _validated_atom(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("{} must be a nonnegative integer".format(field))
    try:
        atom = operator.index(value)
    except TypeError as error:
        raise TypeError("{} must be a nonnegative integer".format(field)) from error
    if atom < 0:
        raise ValueError("{} must be a nonnegative integer".format(field))
    return int(atom)


def _canonical_cell(value: object, period: Cell) -> Cell:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("R must be a tuple of three integers")
    result = []
    for coordinate, extent in zip(value, period):
        if isinstance(coordinate, (bool, np.bool_)):
            raise TypeError("R must contain only integers")
        try:
            integer = operator.index(coordinate)
        except TypeError as error:
            raise TypeError("R must contain only integers") from error
        result.append(int(integer) % extent)
    return result[0], result[1], result[2]


def _validated_blocks(
    blocks: object, period: Cell, rank: int, name: str
) -> List[BlockRecord]:
    if not isinstance(blocks, Mapping):
        raise TypeError("{} blocks must be a mapping".format(name))
    records = []
    for key, array in blocks.items():
        if not isinstance(key, BlockKey):
            raise TypeError("{} block keys must be BlockKey instances".format(name))
        ia1 = _validated_atom(key.ia1, "ia1")
        ia2 = _validated_atom(key.ia2, "ia2")
        cell = _canonical_cell(key.R, period)
        if not isinstance(array, np.ndarray):
            raise TypeError("{} blocks must be NumPy arrays".format(name))
        if array.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
            raise TypeError(
                "{} blocks must have dtype float64 or complex128".format(name)
            )
        if array.ndim != rank:
            raise ValueError("{} blocks must have rank {}".format(name, rank))
        if any(int(extent) <= 0 for extent in array.shape):
            raise ValueError("{} block dimensions must be positive".format(name))
        if not bool(np.all(np.isfinite(array))):
            raise ValueError("{} blocks must contain only finite values".format(name))
        records.append((ia1, ia2, cell, array))
    return records


def _set_dimension(
    dimensions: Dict[int, int], atom: int, size: int, space: str
) -> None:
    if atom in dimensions and dimensions[atom] != size:
        raise ValueError(
            "inconsistent {} dimension for atom {}".format(space, atom)
        )
    dimensions[atom] = int(size)


def _offsets(
    cells: Tuple[Cell, ...], atoms: Tuple[int, ...], dimensions: Mapping[int, int]
) -> Tuple[Dict[OffsetKey, int], int]:
    offsets = {}
    offset = 0
    for cell in cells:
        for atom in atoms:
            offsets[(cell, atom)] = offset
            offset += dimensions[atom]
    return offsets, offset


def infer_supercell_layout(
    coefficient_blocks: Mapping[BlockKey, np.ndarray],
    metric_blocks: Mapping[BlockKey, np.ndarray],
    density_blocks: Mapping[BlockKey, np.ndarray],
    period: Cell,
) -> SupercellLayout:
    """Infer one immutable dense layout and cross-check the C, V, and D maps."""

    checked_period = _validated_period(period)
    coefficient_records = _validated_blocks(
        coefficient_blocks, checked_period, 3, "coefficient"
    )
    metric_records = _validated_blocks(metric_blocks, checked_period, 2, "metric")
    density_records = _validated_blocks(
        density_blocks, checked_period, 2, "density"
    )

    atom_set = set()
    for ia1, ia2, _cell, _array in (
        coefficient_records + metric_records + density_records
    ):
        atom_set.add(ia1)
        atom_set.add(ia2)
    if not atom_set:
        raise ValueError("block maps must contain at least one atom")
    atoms = tuple(sorted(atom_set))
    if any(atom != expected for expected, atom in enumerate(atoms)):
        raise ValueError("atom ids must be contiguous and start at zero")

    ao_dimensions: Dict[int, int] = {}
    auxiliary_dimensions: Dict[int, int] = {}
    for ia1, ia2, _cell, array in coefficient_records:
        _set_dimension(auxiliary_dimensions, ia1, array.shape[0], "auxiliary")
        _set_dimension(ao_dimensions, ia1, array.shape[1], "AO")
        _set_dimension(ao_dimensions, ia2, array.shape[2], "AO")

    for atom in atoms:
        if atom not in ao_dimensions:
            raise ValueError("missing AO dimension for atom {}".format(atom))
        if atom not in auxiliary_dimensions:
            raise ValueError("missing auxiliary dimension for atom {}".format(atom))

    for ia1, ia2, _cell, array in metric_records:
        expected = (auxiliary_dimensions[ia1], auxiliary_dimensions[ia2])
        if array.shape != expected:
            raise ValueError(
                "metric block shape is inconsistent with coefficient dimensions"
            )
    for ia1, ia2, _cell, array in density_records:
        expected = (ao_dimensions[ia1], ao_dimensions[ia2])
        if array.shape != expected:
            raise ValueError(
                "density block shape is inconsistent with coefficient dimensions"
            )

    cells = tuple(
        tuple(int(value) for value in cell) for cell in np.ndindex(checked_period)
    )
    ao_offsets, nao_supercell = _offsets(cells, atoms, ao_dimensions)
    auxiliary_offsets, naux_supercell = _offsets(
        cells, atoms, auxiliary_dimensions
    )
    return SupercellLayout(
        period=checked_period,
        cells=cells,
        atoms=atoms,
        ao_dimensions=ao_dimensions,
        auxiliary_dimensions=auxiliary_dimensions,
        ao_offsets=ao_offsets,
        auxiliary_offsets=auxiliary_offsets,
        nao_supercell=nao_supercell,
        naux_supercell=naux_supercell,
    )


def _validated_max_elements(max_elements: object) -> int:
    if isinstance(max_elements, (bool, np.bool_)):
        raise TypeError("max_elements must be a positive integer")
    try:
        limit = operator.index(max_elements)
    except TypeError as error:
        raise TypeError("max_elements must be a positive integer") from error
    if limit <= 0:
        raise ValueError("max_elements must be a positive integer")
    return int(limit)


def _validated_layout(layout: object) -> SupercellLayout:
    if not isinstance(layout, SupercellLayout):
        raise TypeError("layout must be a SupercellLayout")
    return layout


def _result_dtype(records: List[BlockRecord]):
    if any(array.dtype == np.dtype(np.complex128) for _, _, _, array in records):
        return np.complex128
    return np.float64


def _translated_cell(first: Cell, translation: Cell, period: Cell) -> Cell:
    return (
        (first[0] + translation[0]) % period[0],
        (first[1] + translation[1]) % period[1],
        (first[2] + translation[2]) % period[2],
    )


def _slice(offset: int, dimension: int) -> slice:
    return slice(offset, offset + dimension)


def assemble_pair_coefficient(
    blocks: Mapping[BlockKey, np.ndarray],
    layout: SupercellLayout,
    max_elements: int,
) -> np.ndarray:
    """Expand local-RI C blocks into both ordered AO positions in every cell."""

    checked_layout = _validated_layout(layout)
    limit = _validated_max_elements(max_elements)
    records = _validated_blocks(
        blocks, checked_layout.period, 3, "coefficient"
    )
    for ia1, ia2, _translation, array in records:
        if ia1 not in checked_layout.atoms or ia2 not in checked_layout.atoms:
            raise ValueError("coefficient block refers to unknown atom")
        expected = (
            checked_layout.auxiliary_dimensions[ia1],
            checked_layout.ao_dimensions[ia1],
            checked_layout.ao_dimensions[ia2],
        )
        if array.shape != expected:
            raise ValueError("coefficient block shape is incompatible with layout")

    element_count = (
        int(checked_layout.naux_supercell)
        * int(checked_layout.nao_supercell)
        * int(checked_layout.nao_supercell)
    )
    if element_count > limit:
        raise ValueError("dense supercell allocation exceeds max_elements")
    coefficient = np.zeros(
        (
            checked_layout.naux_supercell,
            checked_layout.nao_supercell,
            checked_layout.nao_supercell,
        ),
        dtype=_result_dtype(records),
    )

    try:
        with np.errstate(over="raise", invalid="raise"):
            for ia1, ia2, translation, block in records:
                for outer_cell in checked_layout.cells:
                    second_cell = _translated_cell(
                        outer_cell, translation, checked_layout.period
                    )
                    auxiliary = _slice(
                        checked_layout.auxiliary_offsets[(outer_cell, ia1)],
                        checked_layout.auxiliary_dimensions[ia1],
                    )
                    first = _slice(
                        checked_layout.ao_offsets[(outer_cell, ia1)],
                        checked_layout.ao_dimensions[ia1],
                    )
                    second = _slice(
                        checked_layout.ao_offsets[(second_cell, ia2)],
                        checked_layout.ao_dimensions[ia2],
                    )
                    coefficient[auxiliary, first, second] += block
                    coefficient[auxiliary, second, first] += block.transpose(
                        0, 2, 1
                    )
    except FloatingPointError as error:
        raise ValueError(
            "coefficient accumulation produced a non-finite value"
        ) from error
    if not bool(np.all(np.isfinite(coefficient))):
        raise ValueError("coefficient accumulation produced a non-finite value")
    return coefficient


def assemble_translation_matrix(
    blocks: Mapping[BlockKey, np.ndarray],
    layout: SupercellLayout,
    space: str,
    max_elements: int,
) -> np.ndarray:
    """Expand one translation-block map into a finite-cell dense matrix."""

    checked_layout = _validated_layout(layout)
    if space not in ("ao", "auxiliary"):
        raise ValueError("space must be 'ao' or 'auxiliary'")
    limit = _validated_max_elements(max_elements)
    records = _validated_blocks(blocks, checked_layout.period, 2, "translation")
    if space == "ao":
        dimensions = checked_layout.ao_dimensions
        offsets = checked_layout.ao_offsets
        total = checked_layout.nao_supercell
    else:
        dimensions = checked_layout.auxiliary_dimensions
        offsets = checked_layout.auxiliary_offsets
        total = checked_layout.naux_supercell

    for ia1, ia2, _translation, array in records:
        if ia1 not in checked_layout.atoms or ia2 not in checked_layout.atoms:
            raise ValueError("translation block refers to unknown atom")
        if array.shape != (dimensions[ia1], dimensions[ia2]):
            raise ValueError("translation block shape is incompatible with layout")

    element_count = int(total) * int(total)
    if element_count > limit:
        raise ValueError("dense supercell allocation exceeds max_elements")
    matrix = np.zeros((total, total), dtype=_result_dtype(records))
    try:
        with np.errstate(over="raise", invalid="raise"):
            for ia1, ia2, translation, block in records:
                for outer_cell in checked_layout.cells:
                    second_cell = _translated_cell(
                        outer_cell, translation, checked_layout.period
                    )
                    first = _slice(offsets[(outer_cell, ia1)], dimensions[ia1])
                    second = _slice(offsets[(second_cell, ia2)], dimensions[ia2])
                    matrix[first, second] += block
    except FloatingPointError as error:
        raise ValueError("matrix accumulation produced a non-finite value") from error
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("matrix accumulation produced a non-finite value")
    return matrix


__all__ = [
    "SupercellLayout",
    "infer_supercell_layout",
    "assemble_pair_coefficient",
    "assemble_translation_matrix",
]
