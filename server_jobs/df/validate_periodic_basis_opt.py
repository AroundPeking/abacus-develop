#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import struct


HEADER = struct.Struct("<16sIIiiiQQ")
MAGIC = b"ABACUS_STBOPT_V1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chunk(path):
    with open(path, "rb") as handle:
        raw_header = handle.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise RuntimeError(f"truncated header: {path}")
        magic, version, kind, iq, ik, ifrequency, rows, columns = HEADER.unpack(raw_header)
        payload = handle.read()
    if magic != MAGIC or version != 1 or len(payload) != 16 * rows * columns:
        raise RuntimeError(f"invalid chunk header or size: {path}")
    values = [complex(real, imag) for real, imag in struct.iter_unpack("<dd", payload)]
    if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in values):
        raise RuntimeError(f"non-finite payload: {path}")
    return {
        "kind": kind,
        "iq": iq,
        "ik": ik,
        "ifrequency": ifrequency,
        "rows": rows,
        "columns": columns,
        "values": values,
    }


def parse_manifest(path):
    metadata = {}
    frequencies = []
    kpoints = []
    eigenvalues = {}
    entries = []
    with open(path, encoding="ascii") as handle:
        first = handle.readline().strip()
        if first != "ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1":
            raise RuntimeError("unexpected manifest version")
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "frequency":
                frequencies.append((int(fields[1]), float(fields[2]), float(fields[3])))
            elif fields[0] == "kpoint":
                occupation_count = int(fields[13])
                occupations = [float(value) for value in fields[14:]]
                if len(occupations) != occupation_count:
                    raise RuntimeError("k-point occupation count mismatch")
                kpoints.append(
                    {
                        "source_ik": int(fields[1]),
                        "target_ik": int(fields[2]),
                        "k_weight": float(fields[12]),
                        "occupations": occupations,
                    }
                )
            elif fields[0] == "eigenvalues_ry":
                source_ik = int(fields[1])
                count = int(fields[2])
                values = [float(value) for value in fields[3:]]
                if len(values) != count or source_ik in eigenvalues:
                    raise RuntimeError("k-point eigenvalue count mismatch or duplicate")
                eigenvalues[source_ik] = values
            elif fields[0] == "entry":
                if len(fields) != 12:
                    raise RuntimeError("manifest entry has the wrong field count")
                entries.append(
                    {
                        "kind": int(fields[1]),
                        "iq": int(fields[2]),
                        "ik": int(fields[3]),
                        "ifrequency": int(fields[4]),
                        "rows": int(fields[5]),
                        "columns": int(fields[6]),
                        "q_weight": float(fields[7]),
                        "k_weight": float(fields[8]),
                        "frequency": float(fields[9]),
                        "path": fields[10],
                        "sha256": fields[11],
                    }
                )
            else:
                metadata[fields[0]] = fields[1:]
    for record in kpoints:
        record["eigenvalues_ry"] = eigenvalues.get(record["source_ik"], [])
        if len(record["eigenvalues_ry"]) != len(record["occupations"]):
            raise RuntimeError("occupied eigenvalue and occupation dimensions differ")
    return metadata, frequencies, kpoints, entries


def hermitian_relative_error(values, dimension):
    scale = max(1.0, max(abs(value) for value in values))
    error = 0.0
    for row in range(dimension):
        for column in range(dimension):
            error = max(
                error,
                abs(values[row * dimension + column]
                    - values[column * dimension + row].conjugate()),
            )
    return error / scale


def sampled_whitening_error(metric, transform, raw_dimension, retained_rank):
    vectors = []
    for index in sorted({0, retained_rank // 2, retained_rank - 1}):
        vector = [0j] * retained_rank
        vector[index] = 1.0 + 0j
        vectors.append(vector)
    vectors.append(
        [complex(math.sin(index + 1.0), math.cos(2.0 * index + 1.0)) for index in range(retained_rank)]
    )
    maximum = 0.0
    for vector in vectors:
        raw_vector = [
            sum(transform[row * retained_rank + column] * vector[column]
                for column in range(retained_rank))
            for row in range(raw_dimension)
        ]
        metric_vector = [
            sum(metric[row * raw_dimension + column] * raw_vector[column]
                for column in range(raw_dimension))
            for row in range(raw_dimension)
        ]
        reconstructed = [
            sum(transform[row * retained_rank + column].conjugate() * metric_vector[row]
                for row in range(raw_dimension))
            for column in range(retained_rank)
        ]
        norm = max(1.0, max(abs(value) for value in vector))
        maximum = max(
            maximum,
            max(abs(reconstructed[index] - vector[index]) for index in range(retained_rank)) / norm,
        )
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    manifest_path = os.path.join(args.dataset, "manifest.dat")
    status_path = os.path.join(args.dataset, "status.dat")
    metadata, frequencies, kpoints, entries = parse_manifest(manifest_path)
    if metadata["abacus_commit"][0] != args.commit:
        raise RuntimeError("ABACUS commit provenance mismatch")
    if int(metadata["entry_count"][0]) != len(entries):
        raise RuntimeError("manifest entry count mismatch")
    expected_entries = 2 + len(frequencies) + len(kpoints) * (4 + len(frequencies))
    if len(entries) != expected_entries:
        raise RuntimeError("dataset is missing an expected global, k-resolved, or frequency-resolved chunk")
    if abs(sum(record["k_weight"] for record in kpoints) - 2.0) > 1.0e-12:
        raise RuntimeError("ABACUS non-spin-polarized full-k weights do not sum to two")

    chunks = {}
    for entry in entries:
        path = os.path.join(args.dataset, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"chunk SHA256 mismatch: {entry['path']}")
        chunk = read_chunk(path)
        for key in ("kind", "iq", "ik", "ifrequency", "rows", "columns"):
            if chunk[key] != entry[key]:
                raise RuntimeError(f"chunk header mismatch: {entry['path']} {key}")
        chunks[(entry["kind"], entry["ik"], entry["ifrequency"])] = chunk

    raw_dimension = int(metadata["raw_auxiliary_dimension"][0])
    retained_rank = int(metadata["whitened_auxiliary_rank"][0])
    primitive_count = int(metadata["primitive_count"][0])
    metric = chunks[(4, 0, -1)]["values"]
    transform = chunks[(5, 0, -1)]["values"]
    metric_hermitian_error = hermitian_relative_error(metric, raw_dimension)
    whitening_error = sampled_whitening_error(metric, transform, raw_dimension, retained_rank)
    declared_whitening_error = float(metadata["coulomb_max_orthonormality_error"][0])
    whitening_probe_limit = max(
        1.0e-8,
        math.sqrt(retained_rank) * declared_whitening_error + 1.0e-12,
    )
    if (metric_hermitian_error > 1.0e-10
            or declared_whitening_error > 1.0e-8
            or whitening_error > whitening_probe_limit):
        raise RuntimeError("full-Coulomb whitening gate failed")

    reference_response_hermitian_error = 0.0
    for ifrequency, _, _ in frequencies:
        response = chunks[(8, 0, ifrequency)]
        if response["rows"] != retained_rank or response["columns"] != retained_rank:
            raise RuntimeError("exact reference-response dimension mismatch")
        reference_response_hermitian_error = max(
            reference_response_hermitian_error,
            hermitian_relative_error(response["values"], retained_rank),
        )
    if reference_response_hermitian_error > 1.0e-10:
        raise RuntimeError("exact reference response is non-Hermitian")

    overlap_hermitian_error = 0.0
    hamiltonian_hermitian_error = 0.0
    kpoint_by_index = {record["source_ik"]: record for record in kpoints}
    for record in kpoints:
        ik = record["source_ik"]
        overlap = chunks[(1, ik, -1)]
        source = chunks[(2, ik, -1)]
        hamiltonian = chunks[(6, ik, -1)]
        occupied_projection = chunks[(7, ik, -1)]
        target = kpoint_by_index[record["target_ik"]]
        if overlap["rows"] != primitive_count or overlap["columns"] != primitive_count:
            raise RuntimeError("overlap dimension mismatch")
        if source["columns"] != primitive_count or source["rows"] != len(record["occupations"]) * retained_rank:
            raise RuntimeError("source dimension mismatch")
        if hamiltonian["rows"] != primitive_count or hamiltonian["columns"] != primitive_count:
            raise RuntimeError("Hamiltonian dimension mismatch")
        if (occupied_projection["rows"] != len(target["occupations"])
                or occupied_projection["columns"] != primitive_count):
            raise RuntimeError("occupied-projection dimension mismatch")
        overlap_hermitian_error = max(
            overlap_hermitian_error,
            hermitian_relative_error(overlap["values"], primitive_count),
        )
        hamiltonian_hermitian_error = max(
            hamiltonian_hermitian_error,
            hermitian_relative_error(hamiltonian["values"], primitive_count),
        )
        for ifrequency, _, _ in frequencies:
            response = chunks[(3, ik, ifrequency)]
            if response["rows"] != source["rows"] or response["columns"] != primitive_count:
                raise RuntimeError("response dimension mismatch")
    if overlap_hermitian_error > 1.0e-10:
        raise RuntimeError("Bloch primitive overlap is non-Hermitian")
    if hamiltonian_hermitian_error > 1.0e-8:
        raise RuntimeError("Bloch primitive Hamiltonian is non-Hermitian")

    with open(status_path, encoding="ascii") as handle:
        status = dict(line.strip().split(maxsplit=1) for line in handle if line.strip())
    if status.get("status") != "success" or status.get("all_converged") != "yes":
        raise RuntimeError("dataset status is not converged success")
    if status.get("physics_hash") != metadata["physics_hash"][0]:
        raise RuntimeError("dataset status and manifest physics hashes differ")

    print(json.dumps({
        "status": "success",
        "entries": len(entries),
        "kpoints": len(kpoints),
        "frequencies": len(frequencies),
        "raw_auxiliary_dimension": raw_dimension,
        "whitened_auxiliary_rank": retained_rank,
        "primitive_count": primitive_count,
        "metric_hermitian_relative_error": metric_hermitian_error,
        "declared_whitening_max_error": declared_whitening_error,
        "sampled_whitening_max_error": whitening_error,
        "sampled_whitening_limit": whitening_probe_limit,
        "reference_response_hermitian_relative_error": reference_response_hermitian_error,
        "overlap_hermitian_relative_error": overlap_hermitian_error,
        "hamiltonian_hermitian_relative_error": hamiltonian_hermitian_error,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
