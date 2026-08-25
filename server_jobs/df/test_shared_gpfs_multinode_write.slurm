#!/bin/bash
#SBATCH --job-name=gpfs_write_probe
#SBATCH --partition=p1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=00:10:00

set -euo pipefail

root=${GPFS_PROBE_ROOT:?set GPFS_PROBE_ROOT to a shared directory}
work=$root/job${SLURM_JOB_ID}
mkdir -p "$work"
export work

srun --nodes=2 --ntasks=2 bash -lc '
  set -euo pipefail
  file="$work/rank-${SLURM_PROCID}-$(hostname).bin"
  dd if=/dev/zero of="$file" bs=1M count=8 conv=fsync status=none
  sha256sum "$file"
'

test "$(find "$work" -maxdepth 1 -type f -name 'rank-*.bin' | wc -l)" -eq 2
test "$(find "$work" -maxdepth 1 -type f -name 'rank-*.bin' -size 8M | wc -l)" -eq 2
echo "SHARED_GPFS_MULTINODE_WRITE_OK work=$work"
