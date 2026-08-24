#!/bin/bash
#SBATCH --job-name=st_basisopt_t4r
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825/logs/regression-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825/logs/regression-%j.err

set -euo pipefail
trap 'status=$?; echo "UNIFIED_BASIS_OPT_STAGE4_REGRESSION_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load cmake/3.31.7
module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825
source_dir=$root/source-staged-af9f01fe2
build_dir=$root/build-3109685
source_commit=60c46b81f5861b7eac9245063adeabfe9e9802a9
result_dir=$root/results/job$SLURM_JOB_ID

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(cat "$root/SOURCE_COMMIT")" = "$source_commit"
test -d "$source_dir"
test -d "$build_dir"
test ! -e "$result_dir"
mkdir -p "$result_dir"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export I_MPI_FABRICS=shm:ofi

cmake --build "$build_dir" --target MODULE_RI_sternheimer_siab_writer_test -j "$SLURM_CPUS_PER_TASK"
ctest --test-dir "$build_dir" --output-on-failure \
  -R '^MODULE_RI_sternheimer_siab_writer_test$' | tee "$result_dir/ctest.log"
printf "status=success\nsource_commit=%s\nbase_build_job=3109685\nslurm_job_id=%s\n" \
  "$source_commit" "$SLURM_JOB_ID" > "$result_dir/provenance.txt"
