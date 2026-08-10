"""Read and write the portable EXXCMP1 v1 tensor snapshot format."""

from __future__ import annotations

import operator
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Mapping, Union

import numpy as np


@dataclass(frozen=True)
class BlockKey:
    ia1: int
    ia2: int
    R: tuple[int, int, int]


if sys.version_info >= (3, 9):
    TensorMap = dict[BlockKey, np.ndarray]
else:  # pragma: no cover - exercised only by the supported Python 3.8 runtime
    TensorMap = Dict[BlockKey, np.ndarray]


@dataclass(frozen=True)
class Snapshot:
    """One EXXCMP1 file, including its rank and scalar-type metadata."""

    version: int
    scalar: str
    rank: int
    nranks: int
    blocks: TensorMap


PathLike = Union[str, Path]

_MAGIC = b"EXXCMP1\0"
_VERSION = 1
_HEADER = struct.Struct("<8sIB3sIIQ")
_KEY_AND_NDIM = struct.Struct("<iiiiiI")
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_SCALAR_TO_CODE = {"real64": 1, "complex128": 2}
_SCALAR_TO_DTYPE = {
    "real64": np.dtype(np.float64),
    "complex128": np.dtype(np.complex128),
}


def _read_exact(input_file: BinaryIO, count: int) -> bytes:
    data = input_file.read(count)
    if len(data) != count:
        raise ValueError("truncated EXX compression snapshot")
    return data


def _read_struct(input_file: BinaryIO, layout: struct.Struct) -> tuple[object, ...]:
    return layout.unpack(_read_exact(input_file, layout.size))


def _read_uint64(input_file: BinaryIO) -> int:
    return int(_read_struct(input_file, struct.Struct("<Q"))[0])


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for extent in shape:
        if extent == 0:
            raise ValueError("tensor shape extents must be nonzero")
        if product > _UINT64_MAX // extent:
            raise ValueError("tensor shape product overflows uint64")
        product *= extent
    return product


def read_snapshot(path: PathLike) -> Snapshot:
    """Read an EXXCMP1 v1 file and return its metadata and tensor blocks."""

    with open(Path(path), "rb", buffering=0) as input_file:
        file_size = os.fstat(input_file.fileno()).st_size
        header_values = _read_struct(input_file, _HEADER)
        magic, version, scalar_code, reserved, rank, nranks, records = header_values
        if magic != _MAGIC:
            raise ValueError("invalid EXX compression snapshot magic")
        if version != _VERSION:
            raise ValueError("unsupported EXX compression snapshot version")
        if scalar_code not in (1, 2):
            raise ValueError("unsupported EXX compression snapshot scalar type")
        if reserved != b"\0\0\0":
            raise ValueError("nonzero reserved EXX compression snapshot header byte")
        if nranks == 0 or rank >= nranks:
            raise ValueError("invalid EXX compression snapshot rank metadata")

        scalar = "real64" if scalar_code == 1 else "complex128"
        scalar_dtype = np.dtype("<f8" if scalar_code == 1 else "<c16")
        bytes_per_value = scalar_dtype.itemsize
        blocks: TensorMap = {}
        for _ in range(int(records)):
            key_values = _read_struct(input_file, _KEY_AND_NDIM)
            ia1, ia2, r0, r1, r2, ndim = key_values
            if ndim < 1 or ndim > 4:
                raise ValueError("tensor ndim must be in [1, 4]")

            shape_values = []
            for _dimension in range(int(ndim)):
                extent = _read_uint64(input_file)
                if extent > sys.maxsize:
                    raise ValueError("tensor shape extent does not fit platform size")
                shape_values.append(extent)
            shape = tuple(shape_values)
            product = _shape_product(shape)
            value_count = _read_uint64(input_file)
            if value_count != product:
                raise ValueError("tensor value count does not match its shape")
            if value_count > _UINT64_MAX // bytes_per_value:
                raise ValueError("tensor byte count overflows uint64")
            if value_count > sys.maxsize:
                raise ValueError("tensor value count does not fit platform size")
            byte_count = value_count * bytes_per_value
            remaining = file_size - input_file.tell()
            if remaining < 0 or byte_count > remaining:
                raise ValueError("truncated EXX compression snapshot tensor data")

            values = np.fromfile(input_file, dtype=scalar_dtype, count=value_count)
            if values.size != value_count:
                raise ValueError("truncated EXX compression snapshot tensor data")
            native_dtype = np.float64 if scalar_code == 1 else np.complex128
            values = values.astype(native_dtype, copy=False)
            array = values.reshape(shape, order="C")

            key = BlockKey(int(ia1), int(ia2), (int(r0), int(r1), int(r2)))
            if key in blocks:
                raise ValueError("duplicate EXX compression snapshot record key")
            blocks[key] = array

        if input_file.read(1) != b"":
            raise ValueError("trailing data in EXX compression snapshot")
    return Snapshot(_VERSION, scalar, int(rank), int(nranks), blocks)


def _checked_integer(value: object, minimum: int, maximum: int, field: str) -> int:
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError("{} must be an integer".format(field)) from error
    if integer < minimum or integer > maximum:
        raise ValueError("{} is outside the supported range".format(field))
    return int(integer)


def _validated_records(blocks: Mapping[BlockKey, np.ndarray], scalar: str):
    expected_dtype = _SCALAR_TO_DTYPE[scalar]
    records = []
    for key, array in blocks.items():
        if not isinstance(key, BlockKey):
            raise TypeError("snapshot block keys must be BlockKey instances")
        if not isinstance(array, np.ndarray):
            raise TypeError("snapshot blocks must be NumPy arrays")
        if array.dtype != expected_dtype:
            raise TypeError(
                "all snapshot blocks must have dtype {}".format(expected_dtype.name)
            )
        if array.ndim < 1 or array.ndim > 4:
            raise ValueError("tensor ndim must be in [1, 4]")
        shape = tuple(int(extent) for extent in array.shape)
        _shape_product(shape)

        ia1 = _checked_integer(key.ia1, _INT32_MIN, _INT32_MAX, "ia1")
        ia2 = _checked_integer(key.ia2, _INT32_MIN, _INT32_MAX, "ia2")
        if not isinstance(key.R, tuple) or len(key.R) != 3:
            raise ValueError("R must be a tuple of three integers")
        cell = tuple(
            _checked_integer(value, _INT32_MIN, _INT32_MAX, "R") for value in key.R
        )
        records.append((ia1, ia2, cell, array))
    records.sort(key=lambda record: (record[0], record[1], record[2]))
    return records


def _write_exact(output_file: BinaryIO, data: object) -> None:
    view = memoryview(data).cast("B")
    while view:
        written = output_file.write(view)
        if written is None or written <= 0:
            raise OSError("failed to write EXX compression snapshot")
        view = view[written:]


def write_snapshot(path: PathLike, snapshot: Snapshot) -> None:
    """Write *snapshot* in deterministic C++-compatible EXXCMP1 v1 layout."""

    if not isinstance(snapshot, Snapshot):
        raise TypeError("snapshot must be a Snapshot instance")
    if snapshot.version != _VERSION:
        raise ValueError("only EXXCMP1 version 1 can be written")
    if snapshot.scalar not in _SCALAR_TO_CODE:
        raise ValueError("scalar must be 'real64' or 'complex128'")
    rank = _checked_integer(snapshot.rank, 0, (1 << 32) - 1, "rank")
    nranks = _checked_integer(snapshot.nranks, 1, (1 << 32) - 1, "nranks")
    if rank >= nranks:
        raise ValueError("invalid EXX compression snapshot rank metadata")
    if not isinstance(snapshot.blocks, Mapping):
        raise TypeError("snapshot blocks must be a mapping")
    records = _validated_records(snapshot.blocks, snapshot.scalar)
    if len(records) > _UINT64_MAX:
        raise ValueError("snapshot record count overflows uint64")

    scalar_code = _SCALAR_TO_CODE[snapshot.scalar]
    scalar_dtype = np.dtype("<f8" if scalar_code == 1 else "<c16")
    destination = Path(path)
    file_descriptor = -1
    temporary_path = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".{}.".format(destination.name), suffix=".tmp", dir=str(destination.parent)
        )
        temporary_path = Path(temporary_name)
        output_file = os.fdopen(file_descriptor, "wb", buffering=1024 * 1024)
        file_descriptor = -1
        with output_file:
            _write_exact(
                output_file,
                _HEADER.pack(
                    _MAGIC, _VERSION, scalar_code, b"\0\0\0", rank, nranks, len(records)
                ),
            )
            for ia1, ia2, cell, array in records:
                _write_exact(
                    output_file,
                    _KEY_AND_NDIM.pack(ia1, ia2, cell[0], cell[1], cell[2], array.ndim),
                )
                _write_exact(output_file, struct.pack("<" + "Q" * array.ndim, *array.shape))
                _write_exact(output_file, struct.pack("<Q", array.size))
                serialized = np.asarray(array, dtype=scalar_dtype, order="C")
                _write_exact(output_file, serialized)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(str(temporary_path), str(destination))
        temporary_path = None
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
