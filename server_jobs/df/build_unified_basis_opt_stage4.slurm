#!/bin/bash
#SBATCH --job-name=st_basisopt_t4
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825/logs/build-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825/logs/build-%j.err

set -euo pipefail
trap 'status=$?; echo "UNIFIED_BASIS_OPT_STAGE4_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load cmake/3.31.7
module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-af9f01fe2-20260825
base=/data/home/df_iopcas_ghj/app/abacus/abacus-siab-source-c-8cf890e8-20260824
bundle=$root/abacus-unified-basis-opt-af9f01fe2-incremental.bundle
bundle_sha=5477ae1fbebd8d8218833df64fed8f67e8b5405d1862c732582faed1f388124c
source_commit=af9f01fe21ff99dd3d7e42e0bab61db66b9d8afe
source_dir=$root/source-staged-af9f01fe2
build_dir=$root/build-$SLURM_JOB_ID
result_dir=$root/results/job$SLURM_JOB_ID
googletest=/data/home/df_iopcas_ghj/app/abacus/abacus-siab-whitened-bf6fd3081-20260823/build_df_9242_siab/_deps/googletest-src

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test -s "$bundle"
test "$(sha256sum "$bundle" | awk '{print $1}')" = "$bundle_sha"
test -d "$source_dir"
test "$(cat "$root/SOURCE_COMMIT")" = "$source_commit"
test ! -e "$build_dir"
test ! -e "$result_dir"
mkdir -p "$result_dir" "$root/logs"

export CC=/data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiicx
export CXX=/data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiicpx
export FC=/data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiifx
export PKG_CONFIG_PATH=/data/app/libxc/6.2.2-intel/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export I_MPI_FABRICS=shm:ofi

cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MPI=ON \
  -DUSE_OPENMP=ON \
  -DENABLE_LCAO=ON \
  -DENABLE_LIBXC=ON \
  -DENABLE_LIBRI=ON \
  -DENABLE_LIBCOMM=ON \
  -DENABLE_GREENX_MINIMAX=ON \
  -DENABLE_MLALGO=OFF \
  -DENABLE_FFT_TWO_CENTER=ON \
  -DBUILD_TESTING=ON \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$googletest" \
  -DLIBRI_DIR=/data/home/df_iopcas_ghj/app/LibRI_latest \
  -DLIBCOMM_DIR=/data/home/df_iopcas_ghj/app/LibComm_fix_status \
  -DELPA_DIR=/data/home/df_iopcas_ghj/app/elpa-build \
  -DCEREAL_INCLUDE_DIR=/data/home/df_iopcas_ghj/app/cereal-1.3.0/include

targets=(
  abacus_3p
  MODULE_IO_read_item_serial
  MODULE_RI_sternheimer_kq_test
  MODULE_RI_sternheimer_coulomb_whitening_test
  MODULE_RI_sternheimer_coulomb_whitening_complex_test
  MODULE_RI_sternheimer_basis_opt_periodic_test
  MODULE_RI_sternheimer_siab_primitives_test
  MODULE_RI_sternheimer_siab_writer_test
  MODULE_RI_sternheimer_abacus_st_smoke_test
)
/usr/bin/time -v cmake --build "$build_dir" --target "${targets[@]}" -j "$SLURM_CPUS_PER_TASK"

ctest --test-dir "$build_dir" --output-on-failure \
  -R '^(MODULE_IO_read_item_serial|MODULE_RI_sternheimer_(kq|coulomb_whitening|coulomb_whitening_complex|basis_opt_periodic|siab_primitives|siab_writer|abacus_st_smoke)_test|MODULE_RI_sternheimer_siab_primitives_test_mpi2)$' \
  | tee "$result_dir/ctest.log"

cp "$build_dir/abacus_3p" "$result_dir/abacus_3p"
sha256sum "$result_dir/abacus_3p" > "$result_dir/abacus_3p.sha256"
cat > "$result_dir/provenance.txt" <<EOF
status=success
source_commit=$source_commit
source_branch=codex/unified-sternheimer-basis-opt
bundle_sha256=$bundle_sha
compiler=$($CXX --version | head -1)
mpi=$(mpirun --version | head -1)
omp_threads=$OMP_NUM_THREADS
slurm_job_id=$SLURM_JOB_ID
EOF
