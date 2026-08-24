#!/bin/bash
#SBATCH --job-name=st_basisopt_t5i
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825/logs/stage5-integration-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825/logs/stage5-integration-%j.err

set -euo pipefail
trap 'status=$?; echo "UNIFIED_BASIS_OPT_STAGE5_INTEGRATION_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load cmake/3.31.7
module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825
source_dir=$root/source-staged-af9f01fe2
build_dir=$root/build-3109685
result_dir=$root/results/stage5-integration-$SLURM_JOB_ID

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test -d "$source_dir"
test -d "$build_dir"
test ! -e "$result_dir"
mkdir -p "$result_dir"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export I_MPI_FABRICS=shm:ofi

targets=(
  abacus_3p
  MODULE_RI_sternheimer_abfs_perturbation_test
  MODULE_RI_sternheimer_basis_opt_periodic_test
  MODULE_RI_sternheimer_coulomb_whitening_complex_test
  MODULE_RI_sternheimer_kq_test
  MODULE_RI_sternheimer_siab_primitives_test
  MODULE_RI_sternheimer_siab_writer_test
  MODULE_RI_sternheimer_abacus_st_smoke_test
)
/usr/bin/time -v cmake --build "$build_dir" --target "${targets[@]}" -j "$SLURM_CPUS_PER_TASK"
ctest --test-dir "$build_dir" --output-on-failure \
  -R '^MODULE_RI_sternheimer_(abfs_perturbation|basis_opt_periodic|coulomb_whitening_complex|kq|siab_primitives|siab_writer|abacus_st_smoke)_test$' \
  | tee "$result_dir/ctest.log"
cp "$build_dir/abacus_3p" "$result_dir/abacus_3p"
sha256sum "$result_dir/abacus_3p" > "$result_dir/abacus_3p.sha256"
printf "status=success\nslurm_job_id=%s\nsource_snapshot=%s\n" \
  "$SLURM_JOB_ID" "$(cat "$root/STAGE5_INTEGRATION_SNAPSHOT")" > "$result_dir/provenance.txt"
