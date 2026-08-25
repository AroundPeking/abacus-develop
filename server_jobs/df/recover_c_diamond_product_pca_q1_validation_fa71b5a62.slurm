#!/bin/bash
#SBATCH --job-name=c24_pca_q1_validate
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/c24-pca-q1-validate-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/c24-pca-q1-validate-%j.err

set -euo pipefail
trap 'status=$?; echo "C_DIAMOND_PRODUCT_PCA_Q1_VALIDATION_RECOVERY_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825
work=$root/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1
dataset=$work/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
top_status=$work/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_SIAB_STATUS.dat
validator=$root/validate_periodic_basis_opt.py
validator_commit=fa71b5a62dfd4ccb3b57c2907201d8c4bc3adae1
expected_validator_sha=75584eeca1471e1121e1c3de000439029e3917decdf674c29ffa0846054f28e3
source_commit=711af860c125b9757c344a1961b63524c550cfe4
expected_abacus_sha=5a7ab8f2fc14424c8b47c5a9cbe22d1784016140cc9fa302f7e032662eed58b2
orbital_sha256=7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d
pseudopotential_sha256=e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(sha256sum "$validator" | awk '{print $1}')" = "$expected_validator_sha"
test -s "$top_status"
grep -qx 'status success' "$top_status"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"
grep -qx 'frequency_count 6' "$dataset/manifest.dat"
grep -qx 'selected_iq 1' "$dataset/manifest.dat"
test ! -s "$work/BASIS_OPT_VALIDATION.json"
test ! -e "$work/PRODUCT_PCA_PROVENANCE.json"
test ! -e "$work/OUTPUT_SHA256.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

cd "$work"
/usr/bin/time -v -o validation.time \
  python3 "$validator" "$dataset" --commit "$source_commit" \
  | tee BASIS_OPT_VALIDATION.json

python3 - "$dataset/manifest.dat" "$orbital_sha256" "$pseudopotential_sha256" "$expected_abacus_sha" \
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

def scalar(key):
    values = metadata.get(key, [])
    if len(values) != 1:
        raise RuntimeError("missing or non-scalar manifest field: " + key)
    return values[0]

digest = hashlib.sha256()
digest.update(b"ABACUS_STERNHEIMER_AUXILIARY_BASIS_DEFINITION_V1")
digest.update(orbital_sha256.encode("ascii"))
digest.update(struct.pack(">d", 1.0e-4))
digest.update(struct.pack(">d", 4.0))
digest.update(struct.pack(">Q", 0))
expected_auxiliary_sha256 = digest.hexdigest()

expected = {
    "auxiliary_basis_source": "product_pca",
    "auxiliary_basis_sha256": expected_auxiliary_sha256,
    "orbital_sha256": orbital_sha256,
    "pseudopotential_sha256": pseudopotential_sha256,
    "executable_sha256": executable_sha256,
}
for key, value in expected.items():
    if scalar(key) != value:
        raise RuntimeError(key + " mismatch")
raw_dimension = int(scalar("raw_auxiliary_dimension"))
retained_rank = int(scalar("whitened_auxiliary_rank"))
if raw_dimension != 236 or not 0 < retained_rank <= raw_dimension:
    raise RuntimeError("unexpected pure product-PCA dimensions")

print(json.dumps({
    "status": "success",
    "auxiliary_basis_source": scalar("auxiliary_basis_source"),
    "auxiliary_basis_sha256": scalar("auxiliary_basis_sha256"),
    "auxiliary_construction_kmesh_times": 4.0,
    "raw_auxiliary_dimension": raw_dimension,
    "whitened_auxiliary_rank": retained_rank,
}, sort_keys=True))
PY

find "$dataset" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUT_SHA256.txt
{
  echo validation_recovery_job=$SLURM_JOB_ID
  echo validator_commit=$validator_commit
  echo validator_sha256=$expected_validator_sha
  echo original_abacus_job=3118459_1
  echo original_abacus_wall_clock=47:34
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/validation_wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/validation_max_rss_kb=/p' validation.time
  echo status=success
  du -sh "$dataset"
} >> RUN_PROVENANCE.txt

echo "C_DIAMOND_PRODUCT_PCA_Q1_VALIDATION_RECOVERY_OK work=$work"
