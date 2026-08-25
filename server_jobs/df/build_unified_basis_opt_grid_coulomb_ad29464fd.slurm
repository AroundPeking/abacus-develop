#!/bin/bash
#SBATCH --job-name=st_gridv_build
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-grid-coulomb-ad29464fd-20260825/logs/build-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-grid-coulomb-ad29464fd-20260825/logs/build-%j.err

set -euo pipefail
trap 'status=$?; echo "UNIFIED_BASIS_OPT_GRID_COULOMB_BUILD_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load cmake/3.31.7
module load git/2.50.0
module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-grid-coulomb-ad29464fd-20260825
source_dir=$root/source
build_dir=$root/build-$SLURM_JOB_ID
result_dir=$root/artifacts/job$SLURM_JOB_ID
source_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
googletest=/data/home/df_iopcas_ghj/app/abacus/abacus-siab-whitened-bf6fd3081-20260823/build_df_9242_siab/_deps/googletest-src

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(git -C "$source_dir" rev-parse HEAD)" = "$source_commit"
test -z "$(git -C "$source_dir" status --short)"
test ! -e "$build_dir"
test ! -e "$result_dir"
mkdir -p "$build_dir" "$result_dir" "$root/logs"

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
  -DDEBUG_INFO=ON \
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
  MODULE_RI_librpa_bz_sampling_test
  MODULE_RI_sternheimer_abfs_perturbation_test
  MODULE_RI_sternheimer_basis_opt_periodic_test
  MODULE_RI_sternheimer_coulomb_whitening_complex_test
  MODULE_RI_sternheimer_kq_test
  MODULE_RI_sternheimer_rpa_test
  MODULE_RI_sternheimer_siab_overlap_test
  MODULE_RI_sternheimer_siab_primitives_test
  MODULE_RI_sternheimer_siab_provenance_test
  MODULE_RI_sternheimer_siab_writer_test
  MODULE_RI_sternheimer_abacus_st_smoke_test
)
/usr/bin/time -v cmake --build "$build_dir" --target "${targets[@]}" -j "$SLURM_CPUS_PER_TASK"
ctest --test-dir "$build_dir" --output-on-failure \
  -R '^(MODULE_IO_read_item_serial|MODULE_RI_librpa_bz_sampling_test|MODULE_RI_sternheimer_(abfs_perturbation|basis_opt_periodic|coulomb_whitening_complex|kq|rpa|siab_overlap|siab_primitives|siab_provenance|siab_writer|abacus_st_smoke)_test)$' \
  | tee "$result_dir/ctest.log"

cp "$build_dir/abacus_3p" "$result_dir/abacus_3p"
sha256sum "$result_dir/abacus_3p" > "$result_dir/abacus_3p.sha256"
grep -F "$source_commit" "$build_dir/commit.h" > "$result_dir/commit_metadata.txt"
grep -E '^(DEBUG_INFO|ENABLE_LIBCOMM|ENABLE_LIBRI):' "$build_dir/CMakeCache.txt" > "$result_dir/cmake_feature_cache.txt"
cat > "$result_dir/provenance.txt" <<EOF
status=success
source_commit=$source_commit
source_branch=codex/unified-sternheimer-basis-opt
compiler=$($CXX --version | head -1)
mpi=$(mpirun --version | head -1)
omp_threads=$OMP_NUM_THREADS
slurm_job_id=$SLURM_JOB_ID
EOF
