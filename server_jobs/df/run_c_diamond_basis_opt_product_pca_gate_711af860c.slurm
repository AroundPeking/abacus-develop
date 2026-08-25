#!/bin/bash
#SBATCH --job-name=c_pca_basis_gate
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/gate-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/gate-%j.err

set -euo pipefail
trap 'status=$?; echo "C_DIAMOND_PRODUCT_PCA_GATE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825
build_job=3117942
abacus=$root/artifacts/job$build_job/abacus_3p
source_commit=711af860c125b9757c344a1961b63524c550cfe4
executable_sha256=5a7ab8f2fc14424c8b47c5a9cbe22d1784016140cc9fa302f7e032662eed58b2
orbital_sha256=7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d
pseudopotential_sha256=e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99
validator_sha256=8e482ec829c93d492ade5c633a224a61c53a390a1e54043bcf0f52e6080880d6
work=$root/gates/c-diamond-product-pca-q1-nfreq1-job$SLURM_JOB_ID

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$executable_sha256"
test "$(sha256sum "$root/validate_periodic_basis_opt.py" | awk '{print $1}')" = "$validator_sha256"
grep -Eq 'double kmesh_times[[:space:]]*=[[:space:]]*4;' \
  "$root/source/source/source_hamilt/module_xc/exx_info.h"
test ! -e "$work"
mkdir -p "$work"
cp -a "$root/gate-template-product-pca/." "$work/"
cd "$work"

test "$(sha256sum C_gga_10au_100Ry_3s3p2d.orb | awk '{print $1}')" = "$orbital_sha256"
test "$(sha256sum C_ONCV_PBE-1.0.upf | awk '{print $1}')" = "$pseudopotential_sha256"
! grep -q '^ABFS_ORBITAL' STRU
test -z "$(find . -maxdepth 1 -type f -name '*.abfs' -print -quit)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_FABRICS=shm:ofi
unset ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL

sha256sum "$abacus" INPUT STRU KPT frequency_grid_nfreq1.dat \
  C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_3s3p2d.orb \
  > RUN_INPUT_SHA256.txt

/usr/bin/time -v "$abacus" > abacus.out 2> abacus.time

status=OUT.C_DIAMOND_BASIS_OPT_PRODUCT_PCA_GATE/STERNHEIMER_SIAB_STATUS.dat
dataset=OUT.C_DIAMOND_BASIS_OPT_PRODUCT_PCA_GATE/STERNHEIMER_BASIS_OPT_V1
test -s "$status"
grep -qx 'status success' "$status"
grep -qx 'format basis_opt_v1' "$status"
grep -qx 'abfs_source product_pca' "$status"
grep -qx 'pca_threshold 0.0001' "$status"
test -s "$dataset/manifest.dat"
test -s "$dataset/status.dat"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"

python3 "$root/validate_periodic_basis_opt.py" \
  "$dataset" --commit "$source_commit" | tee BASIS_OPT_VALIDATION.json

python3 - "$dataset/manifest.dat" "$orbital_sha256" "$pseudopotential_sha256" "$executable_sha256" \
  > PRODUCT_PCA_PROVENANCE.json <<'PY'
import hashlib
import json
import struct
import sys

manifest_path, orbital_sha256, pseudopotential_sha256, executable_sha256 = sys.argv[1:]
metadata = {}
with open(manifest_path, encoding="ascii") as handle:
    for line in handle:
        fields = line.split()
        if len(fields) >= 2 and fields[0] not in {"frequency", "kpoint", "eigenvalues_ry", "entry"}:
            metadata[fields[0]] = fields[1:]

digest = hashlib.sha256()
digest.update(b"ABACUS_STERNHEIMER_AUXILIARY_BASIS_DEFINITION_V1")
digest.update(orbital_sha256.encode("ascii"))
digest.update(struct.pack(">d", 1.0e-4))
digest.update(struct.pack(">d", 4.0))
digest.update(struct.pack(">Q", 0))
expected_auxiliary_sha256 = digest.hexdigest()

def scalar(key):
    values = metadata.get(key, [])
    if len(values) != 1:
        raise RuntimeError("missing or non-scalar manifest field: " + key)
    return values[0]

if scalar("auxiliary_basis_source") != "product_pca":
    raise RuntimeError("periodic producer did not record pure product-PCA")
if scalar("auxiliary_basis_sha256") != expected_auxiliary_sha256:
    raise RuntimeError("independent product-PCA definition hash differs from the manifest")
if scalar("orbital_sha256") != orbital_sha256:
    raise RuntimeError("orbital provenance mismatch")
if scalar("pseudopotential_sha256") != pseudopotential_sha256:
    raise RuntimeError("pseudopotential provenance mismatch")
if scalar("executable_sha256") != executable_sha256:
    raise RuntimeError("executable provenance mismatch")
raw_dimension = int(scalar("raw_auxiliary_dimension"))
retained_rank = int(scalar("whitened_auxiliary_rank"))
if not (0 < retained_rank <= raw_dimension < 706):
    raise RuntimeError("pure product-PCA auxiliary dimension did not decrease from the old stacked value")

print(json.dumps({
    "status": "success",
    "auxiliary_basis_source": scalar("auxiliary_basis_source"),
    "auxiliary_basis_sha256": scalar("auxiliary_basis_sha256"),
    "expected_auxiliary_basis_sha256": expected_auxiliary_sha256,
    "auxiliary_construction_kmesh_times": 4.0,
    "raw_auxiliary_dimension": raw_dimension,
    "whitened_auxiliary_rank": retained_rank,
}, sort_keys=True))
PY

find "$dataset" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUT_SHA256.txt
