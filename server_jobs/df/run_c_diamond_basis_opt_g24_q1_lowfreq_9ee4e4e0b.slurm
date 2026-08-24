#!/bin/bash
#SBATCH --job-name=c24_pi_q1_low
#SBATCH --partition=p1
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/c24-pi-q1-low-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/c24-pi-q1-low-%j.err

set -euo pipefail
trap 'status=$?; echo "C_DIAMOND_G24_Q1_LOWFREQ_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825
build_job=3110409
abacus=$root/artifacts/job$build_job/abacus_3p
source_commit=9ee4e4e0b72dd1c9441a688fb57712f714e33e5b
case_root=/work/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825
work=$case_root/c-diamond-g24-fd8-q1-lowfreq-9ee4e4e0b

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 16
test "$SLURM_NTASKS_PER_NODE" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -x "$abacus"
grep -Fq "$source_commit" "$root/artifacts/job$build_job/commit.h"
test ! -e "$work"
mkdir -p "$work"
cp -a "$root/gate-template/." "$work/"
cd "$work"

set_input_key() {
  local key=$1 value=$2
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' INPUT; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" INPUT
  else
    printf '%-36s %s\n' "$key" "$value" >> INPUT
  fi
}

sed -i -E 's|^[[:space:]]*2[[:space:]]+2[[:space:]]+2[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]*$|4 4 4 0 0 0|' KPT
set_input_key suffix C_DIAMOND_G24_FD8_Q1_LOWFREQ
set_input_key ecutwfc 45
set_input_key nx 24
set_input_key ny 24
set_input_key nz 24
set_input_key kpar 16
set_input_key sternheimer_fd_order 8
set_input_key sternheimer_nfreq 1
set_input_key sternheimer_frequency_mpi 0
set_input_key sternheimer_channel_mpi 0
set_input_key sternheimer_q_index 1
set_input_key exx_pca_threshold 1e-4
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 2
printf '%s\n' '# omega_Ha weight_Ha' \
  '0.043870427724643418 0.089170513957581465' > frequency_grid_nfreq1.dat

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_STACKSIZE=2G
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export I_MPI_FABRICS=shm:ofi
export ABACUS_STERNHEIMER_CHANNEL_THREADS=$OMP_NUM_THREADS
unset ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL

{
  echo status=running
  echo purpose=periodic_c_basis_opt_mother_space_low_frequency_gate
  echo abacus_commit=$source_commit
  echo build_job=$build_job
  echo grid=24,24,24
  echo ecut_ry=45
  echo fd_order=8
  echo kmesh=4,4,4
  echo q_index=1
  echo nfreq=1
  echo omega_ha=0.043870427724643418
  echo frequency_weight_ha=0.089170513957581465
  echo pca_threshold=1e-4
  echo exx_ccp_rmesh_times=1
  echo rpa_ccp_rmesh_times=2
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$SLURM_CPUS_PER_TASK
  sha256sum "$abacus" INPUT STRU KPT frequency_grid_nfreq1.dat \
    C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_3s3p2d.orb C_10au_3s3p2d1f1g_1e-4.abfs
} > RUN_PROVENANCE.txt

/usr/bin/time -v -o abacus.time \
  mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" \
  "$abacus" > abacus.out 2>&1

dataset=OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ/STERNHEIMER_BASIS_OPT_V1
test -s OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ/STERNHEIMER_SIAB_STATUS.dat
grep -qx 'status success' OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ/STERNHEIMER_SIAB_STATUS.dat
grep -qx 'format basis_opt_v1' OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ/STERNHEIMER_SIAB_STATUS.dat
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
python3 "$root/validate_periodic_basis_opt.py" \
  "$dataset" --commit "$source_commit" | tee BASIS_OPT_VALIDATION.json
find "$dataset" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUT_SHA256.txt
{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p' abacus.time
  du -sh "$dataset"
} >> RUN_PROVENANCE.txt

echo "C_DIAMOND_G24_Q1_LOWFREQ_OK work=$work"
