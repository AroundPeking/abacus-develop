#!/bin/bash
#SBATCH --job-name=c_resp_retry
#SBATCH --partition=p1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SIAB_REFERENCE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

abacus=/data/home/df_iopcas_ghj/app/abacus/abacus-siab-source-c-8cf890e8-20260824/artifacts/job3105610/abacus_3p
work=/data/home/df_iopcas_ghj/gw/260825/c-atom-siab-response-retry-v1-56a2af8c

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_FABRICS=shm:ofi

cd "$work"
test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NNODES" = 16
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_SOURCE_V1.dat
test ! -e OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat
test ! -e OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_STATUS.dat

date
echo "$PWD"
echo "$SLURM_JOB_NODELIST"
echo "$SLURM_JOB_CPUS_PER_NODE"
sha256sum "$abacus" INPUT STRU KPT fixed_frequency_grid_nfreq16.dat \
  C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_3s3p2d.orb C_10au_3s3p2d1f1g_1e-4.abfs

/usr/bin/time -v mpirun -ppn 1 -np 16 "$abacus" \
  > "abacus.${SLURM_JOB_ID}.out" \
  2> "abacus.${SLURM_JOB_ID}.time"

status_file=OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_STATUS.dat
test -s "$status_file"
grep -qx 'status success' "$status_file"
grep -qx 'all_converged yes' "$status_file"
test -s OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_SOURCE_V1.dat
test -s OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat
sha256sum "$status_file" OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_SOURCE_V1.dat \
  OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat > OUTPUT_SHA256.txt
date
