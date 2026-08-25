#!/bin/bash
#SBATCH --job-name=c24_pca_nf6_q
#SBATCH --partition=p1
#SBATCH --nodes=48
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=1,2%1
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/c24-pca-nf6-q-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/c24-pca-nf6-q-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C_DIAMOND_PRODUCT_PCA_G24_NFREQ6_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825
build_job=3117942
abacus=$root/artifacts/job$build_job/abacus_3p
source_commit=711af860c125b9757c344a1961b63524c550cfe4
expected_abacus_sha=5a7ab8f2fc14424c8b47c5a9cbe22d1784016140cc9fa302f7e032662eed58b2
validator_commit=fa71b5a62dfd4ccb3b57c2907201d8c4bc3adae1
expected_validator_sha=75584eeca1471e1121e1c3de000439029e3917decdf674c29ffa0846054f28e3
orbital_sha256=7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d
pseudopotential_sha256=e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99
storage_probe_job=3118401
storage_probe=$root/gates/gpfs-write-probe/job$storage_probe_job
run_tag=${STERNHEIMER_RUN_TAG:-retry1}
case_root=$root/runs/c-periodic-basis-opt-product-pca-20260825/$run_tag
iq=${SLURM_ARRAY_TASK_ID:?missing q index}
work=$case_root/c-diamond-g24-fd8-nfreq6-q${iq}-product-pca-711af860c-kpar8-$run_tag
suffix=C_DIAMOND_G24_FD8_NFREQ6_Q${iq}_PRODUCT_PCA

case "$iq" in
  1|2) purpose=periodic_c_basis_opt_product_pca_multifrequency_q_training ;;
  3) purpose=periodic_c_basis_opt_product_pca_independent_q_holdout ;;
  *) echo "unsupported q index: $iq" >&2; exit 2 ;;
esac
test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 48
test "$SLURM_NTASKS_PER_NODE" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test "$(sha256sum "$root/validate_periodic_basis_opt.py" | awk '{print $1}')" = "$expected_validator_sha"
grep -Fq "$source_commit" "$root/artifacts/job$build_job/commit_metadata.txt"
grep -Eq 'double kmesh_times[[:space:]]*=[[:space:]]*4;' \
  "$root/source/source/source_hamilt/module_xc/exx_info.h"
test "$(find "$storage_probe" -maxdepth 1 -type f -name 'rank-*.bin' -size 8M | wc -l)" -eq 2
test ! -e "$work"
mkdir -p "$work"
cp -a "$root/gate-template-product-pca/." "$work/"
cd "$work"
/usr/lpp/mmfs/bin/mmlsquota -g renxg data > STORAGE_QUOTA_START.txt

set_input_key() {
  local key=$1 value=$2
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' INPUT; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" INPUT
  else
    printf '%-36s %s\n' "$key" "$value" >> INPUT
  fi
}

sed -i -E 's|^[[:space:]]*2[[:space:]]+2[[:space:]]+2[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]*$|4 4 4 0 0 0|' KPT
set_input_key suffix "$suffix"
set_input_key ecutwfc 45
set_input_key nx 24
set_input_key ny 24
set_input_key nz 24
set_input_key kpar 8
set_input_key sternheimer_fd_order 8
set_input_key sternheimer_nfreq 6
set_input_key sternheimer_frequency_mpi 1
set_input_key sternheimer_channel_mpi 0
set_input_key sternheimer_mpi_layout frequency_grouped
set_input_key sternheimer_q_index "$iq"
set_input_key exx_pca_threshold 1e-4
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 2
sed -i -E '/^[[:space:]]*sternheimer_frequency_grid_file[[:space:]]+/d' INPUT

! grep -q '^ABFS_ORBITAL' STRU
test -z "$(find . -maxdepth 1 -type f -name '*.abfs' -print -quit)"
test "$(sha256sum C_gga_10au_100Ry_3s3p2d.orb | awk '{print $1}')" = "$orbital_sha256"
test "$(sha256sum C_ONCV_PBE-1.0.upf | awk '{print $1}')" = "$pseudopotential_sha256"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export OMP_STACKSIZE=2G
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export I_MPI_FABRICS=shm:ofi
export ABACUS_STERNHEIMER_CHANNEL_THREADS=$OMP_NUM_THREADS
unset ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL

{
  echo status=running
  echo purpose=$purpose
  echo abacus_commit=$source_commit
  echo build_job=$build_job
  echo validator_commit=$validator_commit
  echo storage_filesystem=data_gpfs
  echo storage_probe_job=$storage_probe_job
  echo run_tag=$run_tag
  echo grid=24,24,24
  echo ecut_ry=45
  echo fd_order=8
  echo kmesh=4,4,4
  echo response_kpoint_groups=8
  echo q_index=$iq
  echo nfreq=6
  echo frequency_grid=greenx_minimax
  echo auxiliary_basis_source=product_pca
  echo auxiliary_construction_kmesh_times=4
  echo pca_threshold=1e-4
  echo exx_ccp_rmesh_times=1
  echo rpa_ccp_rmesh_times=2
  echo mpi_layout=frequency_grouped
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$SLURM_CPUS_PER_TASK
  echo conservative_runtime_gate_hours=16.65
  sha256sum "$abacus" INPUT STRU KPT \
    C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_3s3p2d.orb
} > RUN_PROVENANCE.txt

/usr/bin/time -v -o abacus.time \
  mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" \
  "$abacus" > abacus.out 2>&1

status=OUT.${suffix}/STERNHEIMER_SIAB_STATUS.dat
dataset=OUT.${suffix}/STERNHEIMER_BASIS_OPT_V1
test -s "$status"
grep -qx 'status success' "$status"
grep -qx 'format basis_opt_v1' "$status"
grep -qx 'abfs_source product_pca' "$status"
grep -qx 'sternheimer_nested_k_frequency_mpi yes' "$status"
grep -qx 'sternheimer_kpoint_groups 8' "$status"
grep -qx 'sternheimer_frequency_groups 6' "$status"
grep -qx 'mpi_ranks 48' "$status"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"
grep -qx 'frequency_count 6' "$dataset/manifest.dat"
grep -qx "selected_iq $iq" "$dataset/manifest.dat"

python3 "$root/validate_periodic_basis_opt.py" \
  "$dataset" --commit "$source_commit" | tee BASIS_OPT_VALIDATION.json

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
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p' abacus.time
  du -sh "$dataset"
} >> RUN_PROVENANCE.txt

echo "C_DIAMOND_PRODUCT_PCA_G24_NFREQ6_OK q=$iq work=$work"
