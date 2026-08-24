#!/bin/bash
#SBATCH --job-name=bessel_jle_green
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/bessel-jle-green-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/bessel-jle-green-%j.err

set -euo pipefail
trap 'status=$?; echo "BESSEL_JLE_GREEN_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825
source_root=$root/source
build_root=$root/build
commit=973db989ff4d800b394600e7d4e9b088778d0354
test_binary=$build_root/source/source_io/test/MODULE_IO_bessel_basis_test

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(cat "$root/SOURCE_COMMIT_BESSEL_GREEN")" = "$commit"
grep -Fq 'GlobalV::MY_RANK == 0' \
  "$source_root/source/source_io/module_bessel/bessel_basis.cpp"

cmake --build "$build_root" --target MODULE_IO_bessel_basis_test -j "$SLURM_CPUS_PER_TASK"
test -x "$test_binary"
"$test_binary" \
  --gtest_filter='TestBesselBasis.NonRootRankDoesNotWriteJleOrb:TestBesselBasis.RootRankWritesJleOrb'

echo "BESSEL_JLE_GREEN_CONFIRMED commit=$commit"
