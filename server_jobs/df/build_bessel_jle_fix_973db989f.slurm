#!/bin/bash
#SBATCH --job-name=build_jle_973d
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/build-jle-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/build-jle-%j.err

set -euo pipefail
trap 'status=$?; echo "BESSEL_JLE_BUILD_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load cmake/3.31.7
module load git/2.50.0
module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825
build_dir=$root/build
result_dir=$root/artifacts/job$SLURM_JOB_ID
source_commit=973db989ff4d800b394600e7d4e9b088778d0354

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$result_dir"
mkdir -p "$result_dir" "$root/logs"

cd "$root/source"
test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --short)"

cd "$build_dir"
cmake "$root/source" -DCMAKE_BUILD_TYPE=Release -DGIT_EXECUTABLE="$(command -v git)"
grep -Fq "$source_commit" "$build_dir/commit.h"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export I_MPI_FABRICS=shm:ofi

targets=(
  abacus_3p
  MODULE_IO_bessel_basis_test
  MODULE_IO_read_item_serial
  MODULE_RI_sternheimer_abfs_perturbation_test
  MODULE_RI_sternheimer_basis_opt_periodic_test
  MODULE_RI_sternheimer_coulomb_whitening_complex_test
  MODULE_RI_sternheimer_kq_test
  MODULE_RI_sternheimer_siab_overlap_test
  MODULE_RI_sternheimer_siab_primitives_test
  MODULE_RI_sternheimer_siab_writer_test
  MODULE_RI_sternheimer_abacus_st_smoke_test
)
/usr/bin/time -v cmake --build "$build_dir" --target "${targets[@]}" -j "$SLURM_CPUS_PER_TASK"
ctest --test-dir "$build_dir" --output-on-failure \
  -R '^(MODULE_IO_(bessel_basis_test|read_item_serial)|MODULE_RI_sternheimer_(abfs_perturbation|basis_opt_periodic|coulomb_whitening_complex|kq|siab_overlap|siab_primitives|siab_writer|abacus_st_smoke)_test)$' \
  | tee "$result_dir/ctest.log"

cp "$build_dir/abacus_3p" "$result_dir/abacus_3p"
sha256sum "$result_dir/abacus_3p" > "$result_dir/abacus_3p.sha256"
cp "$build_dir/commit.h" "$result_dir/commit.h"
cat > "$result_dir/provenance.txt" <<EOF
status=success
source_commit=$source_commit
source_branch=codex/unified-sternheimer-basis-opt
omp_threads=$OMP_NUM_THREADS
slurm_job_id=$SLURM_JOB_ID
EOF
