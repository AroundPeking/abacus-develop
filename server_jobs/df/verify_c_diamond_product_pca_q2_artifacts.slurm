#!/bin/bash
#SBATCH --job-name=c24_pca_q2_gate
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/c24-pca-q2-gate-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/logs/c24-pca-q2-gate-%j.err

set -euo pipefail

root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825
work=$root/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q2-product-pca-711af860c-kpar8-retry1
dataset=$work/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -s "$work/RUN_PROVENANCE.txt"
grep -qx 'status=success' "$work/RUN_PROVENANCE.txt"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"
grep -qx 'frequency_count 6' "$dataset/manifest.dat"
grep -qx 'selected_iq 22' "$dataset/manifest.dat"
grep -q '"status": "success"' "$work/BASIS_OPT_VALIDATION.json"
grep -q '"status": "success"' "$work/PRODUCT_PCA_PROVENANCE.json"
test -s "$work/OUTPUT_SHA256.txt"

echo "C_DIAMOND_PRODUCT_PCA_Q2_ARTIFACT_GATE_OK work=$work"
