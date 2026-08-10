# GaAs k444 first-EXX snapshot case

This directory defines the deterministic GaAs input assets for the Stage A-C
occupied-product EXX-compression feasibility study. Task 1 only records the
case, source-file provenance, and intended execution environment. It does not
build ABACUS, submit a job, or run any calculation.

## Scientific scope

- The primitive zinc-blende cell has one Ga and one As atom.
- The `4 4 4` Gamma-centered mesh contains 64 k points because symmetry is
  disabled.
- The Ga and As TZDP orbitals give an expected total of 82 NAOs.
- `INPUT_pbe` is the low-cost PBE producer input.
- `INPUT_exx_snapshot` selects one separate PBE0 hybrid step from the PBE
  density. It is a fixed-density first-EXX snapshot, not a converged PBE0 total
  energy.
- The original LibRI screening thresholds (`exx_pca_threshold`,
  `exx_c_threshold`, `exx_v_threshold`, and `exx_dm_threshold`) are fixed in
  `INPUT_exx_snapshot`. They are separate from the THC-compression thresholds
  varied by the feasibility study.

## Frozen source and execution identity

The feature source is intentionally separate from the normal production
baseline:

```text
server: server 66
remote source/worktree: /home/ghj/abacus/260810/mps-exx-k444/source
feature branch: codex/mps-exx-k444
starting commit: 72499d1ab1ec8aadbe2739f4ca9cee962f93d486
executable: /home/ghj/abacus/260810/mps-exx-k444/build/abacus_3p
execution: one MPI rank, 48 OpenMP threads
```

The executable timestamp is not build provenance. Prepare the feature build
from a checkout with no tracked-file changes and write a manifest beside the
executable that binds the binary and CMake cache to the exact source commit.
On server 66, in the shell with the validated compiler environment loaded, use:

```bash
project_root=/home/ghj/abacus/260810/mps-exx-k444
source_root="$project_root/source"
build_dir="$project_root/build"
abacus_exe="$build_dir/abacus_3p"
cmake_cache="$build_dir/CMakeCache.txt"
build_manifest="$abacus_exe.provenance"

test -z "$(git -C "$source_root" status --porcelain --untracked-files=no)"
test "$(git -C "$source_root" branch --show-current)" = codex/mps-exx-k444

cmake -S "$source_root" -B "$build_dir" \
  -DCMAKE_CXX_COMPILER=/home/apps/intel20u4/compilers_and_libraries_2020.4.304/linux/bin/intel64/icpc \
  -DMPI_CXX_COMPILER=/home/apps/intel20u4/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin/mpiicpc \
  -DCEREAL_DIR=/home/linpz/software/cereal/cereal-1.3.0 \
  -DCEREAL_INCLUDE_DIR=/home/linpz/software/cereal/cereal-1.3.0/include \
  -DELPA_LIBRARY=/home/linpz/software/elpa/elpa-openmp_2021.11.002/lib/libelpa_openmp.so \
  -DELPA_DIR=/home/linpz/software/elpa/elpa-openmp_2021.11.002 \
  -DELPA_INCLUDE_DIR=/home/linpz/software/elpa/elpa-openmp_2021.11.002/include/elpa_openmp-2021.11.002 \
  -DLIBRI_DIR=/home/ghj/abacus/260810/mps-exx-k444/LibRI-21f92 \
  -DLIBCOMM_DIR=/home/ghj/abacus/250920/LibComm \
  -DLibxc_DIR=/home/ghj/abacus/libxc-6.0.0-build \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/home/ghj/abacus/260807/test-deps/googletest-v1.14.0 \
  -DENABLE_MPI=ON -DENABLE_LCAO=ON -DENABLE_LIBRI=ON \
  -DENABLE_LIBCOMM=ON -DENABLE_MLALGO=OFF -DDEBUG_INFO=ON \
  -DBUILD_TESTING=ON
cmake --build "$build_dir" -j20

test -x "$abacus_exe"
test -f "$cmake_cache"
test -z "$(git -C "$source_root" status --porcelain --untracked-files=no)"

source_commit=$(git -C "$source_root" rev-parse --verify HEAD)
executable_sha256=$(sha256sum "$abacus_exe" | awk '{print $1}')
cmakecache_sha256=$(sha256sum "$cmake_cache" | awk '{print $1}')
built_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
manifest_tmp=$(mktemp "$build_manifest.tmp.XXXXXX")
{
  printf 'SOURCE_COMMIT=%s\n' "$source_commit"
  printf 'EXECUTABLE_SHA256=%s\n' "$executable_sha256"
  printf 'CMAKECACHE_SHA256=%s\n' "$cmakecache_sha256"
  printf 'BUILT_AT_UTC=%s\n' "$built_at_utc"
} > "$manifest_tmp"
chmod 0444 "$manifest_tmp"
mv -f "$manifest_tmp" "$build_manifest"
cat "$build_manifest"
```

The later run is a deliberate feature-branch exception. `run_snapshot.slurm`
does not source or evaluate the manifest. It parses only the four documented
keys and refuses to run if a value is missing, duplicated, malformed, or does
not match the clean tracked source checkout, executable, or CMake cache.

## Pseudopotential and orbital provenance

The local source root is:

```text
/Users/ghj/同步空间/AITP_project/sternheimer_abacus/ABACUS-orbitals/Dojo-NC-SR/
```

The four source files are located below that root at:

```text
Pseudopotential/Ga.upf
Pseudopotential/As.upf
Orbitals_v2.0/Ga_TZDP/Ga_gga_8au_100Ry_3s3p3d2f.orb
Orbitals_v2.0/As_TZDP/As_gga_8au_100Ry_3s3p3d2f.orb
```

Copy these four files into the remote run directory; do not create symbolic or
hard links. This keeps the run self-contained and prevents a later change in a
shared source tree from silently changing the case. Copy all seven case assets
into the same run directory and verify the scientific files before submission:

```bash
sha256sum -c checksums.sha256
```

All four lines must report `OK`. Keep `checksums.sha256` with the run outputs so
the exact pseudopotential/orbital chain remains auditable. Immediately before
launch, `run_snapshot.slurm` installs its colocated `INPUT_exx_snapshot` as
`$PWD/INPUT` and verifies the two files byte-for-byte. A stale `INPUT` or
`INPUT_pbe` therefore cannot select the snapshot job's calculation.

## Scheduler preflight

`run_snapshot.slurm` requests one node, one task, 48 CPUs, and 175000 MB on
partition 740. Live-check the partition before submission:

```bash
sinfo -p 740 -o '%P %a %D %c %m %N'
```

Adjust the explicit `--mem` request if the live node inventory requires it.
Also check that `INPUT`, `STRU`, `KPT`, the four verified scientific files, and
the feature executable and its `.provenance` manifest are present. After
verification, the job copies the manifest to `exxcmp.provenance`, prints it in
the Slurm output, and prints the installed snapshot input hash. Submit the job
from this self-contained case/run directory: the script validates and changes
to `SLURM_SUBMIT_DIR` rather than using Slurm's spooled script location. No
compute or scheduler submission is part of Task 1.
