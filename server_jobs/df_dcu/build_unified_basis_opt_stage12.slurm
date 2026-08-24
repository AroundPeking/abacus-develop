#!/bin/bash
#SBATCH --job-name=st_basisopt_t12
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --output=/work1/ghj/unified-basis-opt-20260825/logs/stage12-build-%j.out
#SBATCH --error=/work1/ghj/unified-basis-opt-20260825/logs/stage12-build-%j.err

set -euo pipefail
trap 'status=$?; echo "UNIFIED_BASIS_OPT_STAGE12_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

source /public/home/ghj/app/src/env_60_245_intel2021.sh

root=/work1/ghj/unified-basis-opt-20260825
archive=$root/archives/abacus-unified-basis-opt-stage12-20260825.tar.gz
archive_sha=770d45943e68a41912f6e8ce4bc8b6096a8835b854e2a331f75a912c69b521c1
source_dir=$root/source-stage12-$SLURM_JOB_ID
build_dir=$root/build-stage12-$SLURM_JOB_ID
result_dir=$root/results/stage12-$SLURM_JOB_ID
googletest=/work1/ghj/delta-st-periodic-fd8-20260822/googletest-periodic-fd8-20260822
libri=/public/home/ghj/app/src/si-solid-delta-20260814/deps/LibRI
libcomm=/public/home/ghj/app/src/si-solid-delta-20260814/deps/LibComm-ring-final

test "$SLURM_JOB_PARTITION" = normal
test -s "$archive"
test "$(sha256sum "$archive" | awk '{print $1}')" = "$archive_sha"
test -f "$googletest/googletest/include/gtest/gtest.h"
test ! -e "$source_dir"
test ! -e "$build_dir"
test ! -e "$result_dir"

mkdir -p "$source_dir" "$result_dir" "$root/logs"
tar -xzf "$archive" -C "$source_dir"

export PATH=/public/home/ghj/.conda/envs/ds092/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export PKG_CONFIG_PATH=/public/software/apps/abacus/libxc-5.2.3/lib/pkgconfig:${PKG_CONFIG_PATH:-}

cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=/public/software/mpi/intelmpi/2021.3.0/bin/mpiicc \
  -DCMAKE_CXX_COMPILER=/public/software/mpi/intelmpi/2021.3.0/bin/mpiicpc \
  -DCMAKE_Fortran_COMPILER=/public/software/mpi/intelmpi/2021.3.0/bin/mpiifort \
  -DPython3_EXECUTABLE=/public/home/ghj/.conda/envs/ds092/bin/python \
  -DENABLE_MPI=ON \
  -DUSE_OPENMP=ON \
  -DENABLE_LCAO=ON \
  -DENABLE_LIBXC=ON \
  -DENABLE_LIBRI=ON \
  -DENABLE_LIBCOMM=ON \
  -DENABLE_GREENX_MINIMAX=ON \
  -DENABLE_MLALGO=OFF \
  -DENABLE_FFT_TWO_CENTER=ON \
  -DDEBUG_INFO=ON \
  -DBUILD_TESTING=ON \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$googletest" \
  -DGIT_SUBMODULE=OFF \
  -DLIBRI_DIR="$libri" \
  -DLIBCOMM_DIR="$libcomm" \
  -DELPA_DIR=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021 \
  -DCEREAL_INCLUDE_DIR=/public/home/ghj/app/deps/cereal-1.3.0/include

targets=(
  MODULE_IO_read_item_serial
  MODULE_RI_sternheimer_coulomb_whitening_test
  MODULE_RI_sternheimer_coulomb_whitening_complex_test
  MODULE_RI_sternheimer_siab_writer_test
)
cmake --build "$build_dir" --target "${targets[@]}" -j "$OMP_NUM_THREADS"

ctest --test-dir "$build_dir" \
  -R '^(MODULE_IO_read_item_serial|MODULE_RI_sternheimer_coulomb_whitening_test|MODULE_RI_sternheimer_coulomb_whitening_complex_test|MODULE_RI_sternheimer_siab_writer_test)$' \
  --output-on-failure | tee "$result_dir/ctest.log"

cat > "$result_dir/provenance.txt" <<EOF
status=success
purpose=unified_sternheimer_basis_optimization_stage12_tests
source_base_commit=56a2af8c7f765ca59ec3ceff0d204c0384fe0a14
source_snapshot_sha256=$archive_sha
build_job_id=$SLURM_JOB_ID
build_host=$(hostname)
build_time=$(date -Iseconds)
partition=$SLURM_JOB_PARTITION
cpus_per_task=$SLURM_CPUS_PER_TASK
memory_mb=110610
tests=input_alias,real_coulomb_whitening,complex_coulomb_whitening,molecular_writer
EOF

echo "UNIFIED_BASIS_OPT_STAGE12_OK result=$result_dir"
