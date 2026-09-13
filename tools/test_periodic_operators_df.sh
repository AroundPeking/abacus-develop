#!/bin/bash
set -euo pipefail
module load oneapi/2024.2
cd "${1:?isolated test source directory required}"
reference=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/build-3117942
gtest=/data/home/df_iopcas_ghj/app/abacus/abacus-siab-whitened-bf6fd3081-20260823/build_df_9242_siab/_deps/googletest-src/googletest
icpx -std=c++17 -pthread -I source -I "$gtest/include" \
  source/source_lcao/module_ri/test/sternheimer_basis_opt_periodic_test.cpp \
  source/source_lcao/module_ri/sternheimer_basis_opt_periodic.cpp \
  source/source_lcao/module_ri/sternheimer_siab_provenance.cpp \
  "$reference/lib/libgtest_main.a" "$reference/lib/libgtest.a" -o contract_test
./contract_test
