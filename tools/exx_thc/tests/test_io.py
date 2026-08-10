from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from exx_thc.io import BlockKey, Snapshot, read_snapshot, write_snapshot


HEADER = struct.Struct("<8sIB3sIIQ")
KEY_AND_NDIM = struct.Struct("<iiiiiI")


def encoded_record(key: BlockKey, array: np.ndarray, scalar_code: int) -> bytes:
    result = bytearray(
        KEY_AND_NDIM.pack(key.ia1, key.ia2, *key.R, array.ndim)
    )
    result += struct.pack("<" + "Q" * array.ndim, *array.shape)
    result += struct.pack("<Q", array.size)
    flat = array.ravel(order="C")
    if scalar_code == 1:
        result += flat.astype("<f8", copy=False).tobytes(order="C")
    else:
        parts = np.empty((flat.size, 2), dtype="<f8")
        parts[:, 0] = flat.real
        parts[:, 1] = flat.imag
        result += parts.tobytes(order="C")
    return bytes(result)


def encoded_snapshot(
    records: list[tuple[BlockKey, np.ndarray]],
    scalar_code: int,
    rank: int = 0,
    nranks: int = 1,
) -> bytes:
    result = bytearray(
        HEADER.pack(b"EXXCMP1\0", 1, scalar_code, b"\0\0\0", rank, nranks, len(records))
    )
    for key, array in records:
        result += encoded_record(key, array, scalar_code)
    return bytes(result)


class SnapshotIOTest(unittest.TestCase):
    def write_bytes(self, data: bytes) -> Path:
        temporary = tempfile.NamedTemporaryFile(delete=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        temporary.write(data)
        temporary.close()
        return Path(temporary.name)

    def test_reads_handcrafted_real_cpp_schema(self) -> None:
        key = BlockKey(0x01020304, 7, (-2, 3, 1))
        array = np.array([[1.0, -2.5, 3.25], [4.5, 5.0, -6.0]], dtype=np.float64)
        snapshot = read_snapshot(self.write_bytes(encoded_snapshot([(key, array)], 1, 2, 5)))

        self.assertEqual(snapshot.version, 1)
        self.assertEqual(snapshot.scalar, "real64")
        self.assertEqual((snapshot.rank, snapshot.nranks), (2, 5))
        np.testing.assert_array_equal(snapshot.blocks[key], array)
        self.assertEqual(snapshot.blocks[key].dtype, np.dtype(np.float64))

    def test_reads_handcrafted_complex_cpp_schema(self) -> None:
        key = BlockKey(-3, 4, (2, -1, 0))
        array = np.array(
            [[[1.0 + 2.0j, -3.0 + 0.5j]], [[4.0 - 5.0j, 6.0 + 0.0j]]],
            dtype=np.complex128,
        )
        snapshot = read_snapshot(self.write_bytes(encoded_snapshot([(key, array)], 2)))

        self.assertEqual(snapshot.scalar, "complex128")
        np.testing.assert_array_equal(snapshot.blocks[key], array)
        self.assertEqual(snapshot.blocks[key].dtype, np.dtype(np.complex128))

    def test_writer_matches_exact_cpp_layout_and_roundtrips(self) -> None:
        real_key = BlockKey(7, 4, (-2, 3, 1))
        real = np.arange(6, dtype=np.float64).reshape(2, 3) - 2.5
        complex_key = BlockKey(1, 0, (2, -1, 0))
        complex_array = np.array(
            [[1.0 + 2.0j, 3.0 - 4.0j, -5.0 + 0.25j], [6.0j, 7.0, -8.0j]],
            dtype=np.complex128,
        )
        cases = [
            (Snapshot(1, "real64", 2, 5, {real_key: real}), real_key, real, 1),
            (
                Snapshot(1, "complex128", 0, 4, {complex_key: complex_array}),
                complex_key,
                complex_array,
                2,
            ),
        ]
        for snapshot, key, expected_array, scalar_code in cases:
            with self.subTest(scalar=snapshot.scalar):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "snapshot.exxcmp"
                    write_snapshot(path, snapshot)
                    expected = encoded_snapshot(
                        [(key, expected_array)], scalar_code, snapshot.rank, snapshot.nranks
                    )
                    self.assertEqual(path.read_bytes(), expected)
                    restored = read_snapshot(path)
                    self.assertEqual(restored.scalar, snapshot.scalar)
                    np.testing.assert_array_equal(restored.blocks[key], expected_array)

    def test_reader_and_writer_use_streams_not_path_whole_file_helpers(self) -> None:
        key = BlockKey(2, 3, (-1, 0, 2))
        array = np.array([[1.0 + 2.0j, -3.0 + 4.0j]], dtype=np.complex128)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.exxcmp"
            destination = Path(directory) / "destination.exxcmp"
            source.write_bytes(encoded_snapshot([(key, array)], 2))

            with mock.patch.object(
                Path, "read_bytes", side_effect=AssertionError("whole-file read used")
            ), mock.patch.object(
                Path, "write_bytes", side_effect=AssertionError("whole-file write used")
            ):
                snapshot = read_snapshot(source)
                write_snapshot(destination, snapshot)

            np.testing.assert_array_equal(snapshot.blocks[key], array)
            self.assertEqual(destination.read_bytes(), encoded_snapshot([(key, array)], 2))

    def test_atomic_writer_preserves_destination_and_removes_temp_on_fsync_error(self) -> None:
        key = BlockKey(0, 0, (0, 0, 0))
        snapshot = Snapshot(1, "real64", 0, 1, {key: np.array([2.0], dtype=np.float64)})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "snapshot.exxcmp"
            original = encoded_snapshot(
                [(key, np.array([9.0], dtype=np.float64))], scalar_code=1
            )
            destination.write_bytes(original)

            with mock.patch("exx_thc.io.os.fsync", side_effect=OSError("injected fsync failure")):
                with self.assertRaises(OSError):
                    write_snapshot(destination, snapshot)

            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_rejects_bad_header_truncation_duplicate_and_trailing_bytes(self) -> None:
        key = BlockKey(0, 0, (0, 0, 0))
        array = np.array([1.0], dtype=np.float64)
        valid = encoded_snapshot([(key, array)], 1)
        record = encoded_record(key, array, 1)
        invalid = {
            "magic": b"X" + valid[1:],
            "version": valid[:8] + struct.pack("<I", 2) + valid[12:],
            "scalar": valid[:12] + b"\x03" + valid[13:],
            "reserved": valid[:13] + b"\x01" + valid[14:],
            "rank": valid[:16] + struct.pack("<II", 1, 1) + valid[24:],
            "truncated": valid[:-1],
            "duplicate": HEADER.pack(b"EXXCMP1\0", 1, 1, b"\0\0\0", 0, 1, 2)
            + record
            + record,
            "trailing": valid + b"\xff",
        }
        for name, data in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_snapshot(self.write_bytes(data))

    def test_rejects_invalid_shape_metadata(self) -> None:
        key = BlockKey(0, 0, (0, 0, 0))
        base = bytearray(encoded_snapshot([(key, np.ones((2, 3), dtype=np.float64))], 1))
        cases: dict[str, bytes] = {}
        bad_ndim = bytearray(base)
        struct.pack_into("<I", bad_ndim, 52, 0)
        cases["ndim"] = bytes(bad_ndim)
        zero_shape = bytearray(base)
        struct.pack_into("<Q", zero_shape, 56, 0)
        cases["zero shape"] = bytes(zero_shape)
        bad_count = bytearray(base)
        struct.pack_into("<Q", bad_count, 72, 5)
        cases["count"] = bytes(bad_count)
        overflow = bytearray(base)
        struct.pack_into("<Q", overflow, 56, (1 << 64) - 1)
        struct.pack_into("<Q", overflow, 64, 2)
        cases["overflow"] = bytes(overflow)

        for name, data in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_snapshot(self.write_bytes(data))

    def test_writer_rejects_type_key_shape_and_metadata_errors(self) -> None:
        key = BlockKey(0, 0, (0, 0, 0))
        valid = np.ones((2, 2), dtype=np.float64)
        cases = {
            "wrong dtype": Snapshot(1, "real64", 0, 1, {key: valid.astype(np.float32)}),
            "declared scalar mismatch": Snapshot(
                1, "real64", 0, 1, {key: valid.astype(np.complex128)}
            ),
            "mixed dtypes": Snapshot(
                1,
                "complex128",
                0,
                1,
                {
                    key: valid.astype(np.complex128),
                    BlockKey(1, 0, (0, 0, 0)): valid,
                },
            ),
            "key narrowing": Snapshot(
                1, "real64", 0, 1, {BlockKey(1 << 31, 0, (0, 0, 0)): valid}
            ),
            "zero extent": Snapshot(
                1, "real64", 0, 1, {key: np.empty((2, 0), dtype=np.float64)}
            ),
            "rank": Snapshot(1, "real64", 1, 1, {key: valid}),
            "version": Snapshot(2, "real64", 0, 1, {key: valid}),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.exxcmp"
            for name, snapshot in cases.items():
                with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                    write_snapshot(path, snapshot)


if __name__ == "__main__":
    unittest.main()
