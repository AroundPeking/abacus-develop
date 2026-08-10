#!/bin/bash
#SBATCH --job-name=gaas_k444_exxcmp
#SBATCH --partition=640
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=175000
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

set -euo pipefail

die()
{
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

submitted_run_dir=${SLURM_SUBMIT_DIR:-$PWD}
[[ -d "$submitted_run_dir" ]] || die "run directory is missing: $submitted_run_dir"
run_dir=$(cd -- "$submitted_run_dir" && pwd -P) \
    || die "cannot resolve run directory: $submitted_run_dir"
cd -- "$run_dir"

source_root=/home/ghj/abacus/260810/mps-exx-k444/source
abacus_exe=/home/ghj/abacus/260810/mps-exx-k444/build/abacus_3p
cmake_cache=/home/ghj/abacus/260810/mps-exx-k444/build/CMakeCache.txt
build_manifest=/home/ghj/abacus/260810/mps-exx-k444/build/abacus_3p.provenance
snapshot_input="$run_dir/INPUT_exx_snapshot"
active_input="$run_dir/INPUT"

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export ABACUS_EXX_COMPRESSION_DUMP="$run_dir/exxcmp"

for required_command in git sha256sum install cmp; do
    command -v "$required_command" >/dev/null 2>&1 \
        || die "required command not found: $required_command"
done

[[ -x "$abacus_exe" ]] || die "executable is missing or not executable: $abacus_exe"
[[ -f "$cmake_cache" ]] || die "CMake cache is missing: $cmake_cache"
[[ -f "$build_manifest" ]] || die "build provenance manifest is missing: $build_manifest"
[[ -f "$snapshot_input" ]] || die "snapshot input is missing: $snapshot_input"

source_commit=
executable_sha256=
cmakecache_sha256=
built_at_utc=
line_number=0
while IFS= read -r manifest_line || [[ -n "$manifest_line" ]]; do
    line_number=$((line_number + 1))
    [[ "$manifest_line" == *=* ]] \
        || die "malformed build manifest line $line_number"
    manifest_key=${manifest_line%%=*}
    manifest_value=${manifest_line#*=}
    [[ -n "$manifest_value" ]] \
        || die "empty build manifest value for $manifest_key"
    case "$manifest_key" in
        SOURCE_COMMIT)
            [[ -z "$source_commit" ]] || die "duplicate SOURCE_COMMIT"
            source_commit=$manifest_value
            ;;
        EXECUTABLE_SHA256)
            [[ -z "$executable_sha256" ]] || die "duplicate EXECUTABLE_SHA256"
            executable_sha256=$manifest_value
            ;;
        CMAKECACHE_SHA256)
            [[ -z "$cmakecache_sha256" ]] || die "duplicate CMAKECACHE_SHA256"
            cmakecache_sha256=$manifest_value
            ;;
        BUILT_AT_UTC)
            [[ -z "$built_at_utc" ]] || die "duplicate BUILT_AT_UTC"
            built_at_utc=$manifest_value
            ;;
        *)
            die "unknown build manifest key: $manifest_key"
            ;;
    esac
done < "$build_manifest"

[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || die "invalid SOURCE_COMMIT"
[[ "$executable_sha256" =~ ^[0-9a-f]{64}$ ]] || die "invalid EXECUTABLE_SHA256"
[[ "$cmakecache_sha256" =~ ^[0-9a-f]{64}$ ]] || die "invalid CMAKECACHE_SHA256"
[[ "$built_at_utc" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] \
    || die "invalid BUILT_AT_UTC"

tracked_status=$(cd -- "$source_root" && git status --porcelain --untracked-files=no)
[[ -z "$tracked_status" ]] || die "source checkout has tracked-file changes"

current_source_commit=$(cd -- "$source_root" && git rev-parse --verify HEAD)
[[ "$current_source_commit" == "$source_commit" ]] \
    || die "source HEAD does not match build manifest"

current_executable_sha256=$(sha256sum "$abacus_exe")
current_executable_sha256=${current_executable_sha256%% *}
[[ "$current_executable_sha256" == "$executable_sha256" ]] \
    || die "executable hash does not match build manifest"

current_cmakecache_sha256=$(sha256sum "$cmake_cache")
current_cmakecache_sha256=${current_cmakecache_sha256%% *}
[[ "$current_cmakecache_sha256" == "$cmakecache_sha256" ]] \
    || die "CMake cache hash does not match build manifest"

verified_manifest_copy="${ABACUS_EXX_COMPRESSION_DUMP}.provenance"
install -m 0444 -- "$build_manifest" "$verified_manifest_copy"
cmp -s -- "$build_manifest" "$verified_manifest_copy" \
    || die "verified manifest copy differs from source"

install -m 0644 -- "$snapshot_input" "$active_input"
cmp -s -- "$snapshot_input" "$active_input" \
    || die "active INPUT differs from INPUT_exx_snapshot"

date
hostname
pwd
printf 'OMP_NUM_THREADS=%s\n' "$OMP_NUM_THREADS"
printf '%s\n' 'Verified build provenance:'
cat -- "$verified_manifest_copy"
printf 'Verified snapshot input SHA256: %s\n' "$(sha256sum "$active_input" | awk '{print $1}')"

"$abacus_exe"

date
