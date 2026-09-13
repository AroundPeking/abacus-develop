#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96000M
#SBATCH --time=02:00:00
set -euo pipefail
root=${1:?isolated root required}
commit=${2:?source commit required}
module load cmake/3.31.7 git/2.50.0 oneapi/2024.2
source_dir=$root/source
build_dir=$root/build
test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$source_dir" status --porcelain)"
mkdir "$root/build-once.lock"
mkdir "$root/result"
status=failed
finish() {
    rc=$?
    trap - EXIT
    printf 'status=%s\nexit_code=%s\njob_id=%s\nsource_commit=%s\n' \
        "$status" "$rc" "$SLURM_JOB_ID" "$commit" > "$root/result/BUILD_STATUS"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export CC=mpiicx CXX=mpiicpx FC=mpiifx
export PKG_CONFIG_PATH=/data/app/libxc/6.2.2-intel/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
gtest=/data/home/df_iopcas_ghj/app/abacus/abacus-siab-whitened-bf6fd3081-20260823/build_df_9242_siab/_deps/googletest-src
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DUSE_OPENMP=ON \
    -DENABLE_LCAO=ON -DENABLE_LIBXC=ON -DENABLE_LIBRI=ON -DENABLE_LIBCOMM=ON \
    -DENABLE_GREENX_MINIMAX=ON -DDEBUG_INFO=ON -DENABLE_MLALGO=OFF \
    -DENABLE_FFT_TWO_CENTER=ON -DBUILD_TESTING=ON \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$gtest" \
    -DLIBRI_DIR=/data/home/df_iopcas_ghj/app/LibRI_latest \
    -DLIBCOMM_DIR=/data/home/df_iopcas_ghj/app/LibComm_fix_status \
    -DELPA_DIR=/data/home/df_iopcas_ghj/app/elpa-build \
    -DCEREAL_INCLUDE_DIR=/data/home/df_iopcas_ghj/app/cereal-1.3.0/include
/usr/bin/time -v cmake --build "$build_dir" --target abacus_3p \
    MODULE_IO_read_item_serial MODULE_RI_sternheimer_basis_opt_periodic_test \
    MODULE_RI_sternheimer_abacus_st_smoke_test -j "$SLURM_CPUS_PER_TASK"
ctest --test-dir "$build_dir" --output-on-failure \
    -R '^(MODULE_IO_read_item_serial|MODULE_RI_sternheimer_(basis_opt_periodic|abacus_st_smoke)_test)$' \
    | tee "$root/result/ctest.log"
test -s "$build_dir/abacus_3p"
grep -F "$commit" "$build_dir/commit.h"
test -z "$(git -C "$source_dir" status --porcelain)"
sha256sum "$build_dir/abacus_3p" "$build_dir/CMakeCache.txt" > "$root/result/SHA256SUMS"
git -C "$source_dir" archive HEAD | sha256sum > "$root/result/SOURCE_ARCHIVE_SHA256"
grep -E '^(ENABLE_MPI|ENABLE_LCAO|ENABLE_LIBRI|ENABLE_LIBCOMM|DEBUG_INFO):BOOL=ON' \
    "$build_dir/CMakeCache.txt" > "$root/result/features.txt"
test "$(wc -l < "$root/result/features.txt")" -eq 5
status=success
