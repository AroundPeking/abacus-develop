#!/bin/bash
#SBATCH --job-name=bessel_jle_red
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/bessel-jle-red-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825/logs/bessel-jle-red-%j.err

set -euo pipefail
trap 'status=$?; echo "BESSEL_JLE_RED_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-1474d95c2-20260825
source_root=$root/source
build_root=$root/build
bundle=$root/9233ad49c.bundle
commit=9233ad49c9cf1cc05936fa7e03751b417e36e15f
test_binary=$build_root/source/source_io/test/MODULE_IO_bessel_basis_test

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_CPUS_PER_TASK" = 40
test -s "$bundle"
test "$(cat "$root/SOURCE_COMMIT_BESSEL_RED")" = "$commit"
grep -Fq 'NonRootRankDoesNotWriteJleOrb' \
  "$source_root/source/source_io/test/bessel_basis_test.cpp"

cmake --build "$build_root" --target MODULE_IO_bessel_basis_test -j "$SLURM_CPUS_PER_TASK"
test -x "$test_binary"
set +e
"$test_binary" --gtest_filter=TestBesselBasis.NonRootRankDoesNotWriteJleOrb \
  > "$root/logs/bessel-jle-red-test-$SLURM_JOB_ID.log" 2>&1
test_status=$?
set -e
test "$test_status" -ne 0
grep -Fq 'NonRootRankDoesNotWriteJleOrb' \
  "$root/logs/bessel-jle-red-test-$SLURM_JOB_ID.log"
grep -Fq 'output.good()' "$root/logs/bessel-jle-red-test-$SLURM_JOB_ID.log"

echo "BESSEL_JLE_RED_CONFIRMED commit=$commit test_status=$test_status"
