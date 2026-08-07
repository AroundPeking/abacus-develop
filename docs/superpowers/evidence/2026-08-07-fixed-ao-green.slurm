#!/bin/bash
#SBATCH --job-name=st-fixedao-green
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=24:00:00

set -euo pipefail

campaign=/work1/ghj/sternheimer_abacus_tests/fixed_ao_galerkin_20260807
archive=${campaign}/input/abacus-fixed-ao-green-ad3d673a1.tar.gz
work=/tmp/st-fixedao-green-${SLURM_JOB_ID}
result=${campaign}/jobs/${SLURM_JOB_ID}

mkdir -p "${work}/source" "${work}/build" "${result}"
trap 'status=$?; cp -a "${work}/." "${result}/" 2>/dev/null || true; exit ${status}' EXIT

source /public/home/ghj/app/src/env_60_245_intel2021.sh
sha256sum "${archive}" > "${work}/archive.sha256"
tar -xzf "${archive}" -C "${work}/source"

cmake -S "${work}/source" -B "${work}/build" \
  -DENABLE_MPI=ON \
  -DENABLE_LCAO=ON \
  -DENABLE_LIBRI=ON \
  -DENABLE_LIBCOMM=ON \
  -DENABLE_GREENX_MINIMAX=ON \
  -DENABLE_MLALGO=OFF \
  -DDEBUG_INFO=ON \
  -DBUILD_TESTING=ON \
  -DLIBRI_DIR=/public/home/ghj/app/src/master_ghj_ewald_20260608/src/LibRI_latest \
  -DLIBCOMM_DIR=/public/home/ghj/app/src/master_ghj_ewald_20260608/src/LibComm_fix_status \
  -DCEREAL_INCLUDE_DIR=/public/home/ghj/app/deps/cereal-1.3.0/include \
  -DELPA_DIR=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021 \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/work1/ghj/sternheimer_abacus_tests/siab_full_integration_v9_20260721/deps/googletest-v1.14.0 \
  > "${work}/configure.stdout" 2> "${work}/configure.stderr"

cmake --build "${work}/build" -j30 --target \
  MODULE_RI_sternheimer_siab_overlap_test \
  MODULE_RI_sternheimer_siab_fixed_ao_test \
  MODULE_RI_sternheimer_siab_writer_test \
  MODULE_RI_sternheimer_abacus_st_smoke_test \
  > "${work}/build-tests.stdout" 2> "${work}/build-tests.stderr"

ctest --test-dir "${work}/build" --output-on-failure -R \
  'MODULE_RI_sternheimer_(siab_overlap|siab_fixed_ao|siab_writer|abacus_st_smoke)_test$' \
  > "${work}/ctest.stdout" 2> "${work}/ctest.stderr"

cmake --build "${work}/build" -j30 --target abacus_3p \
  > "${work}/build-abacus.stdout" 2> "${work}/build-abacus.stderr"

grep -E 'ENABLE_(LIBRI|LIBCOMM|MPI|LCAO)|DEBUG_INFO' "${work}/build/CMakeCache.txt" \
  > "${work}/feature-cache.txt"
sha256sum "${work}/build/abacus_3p" > "${work}/executable.sha256"
touch "${work}/completed.ok"
