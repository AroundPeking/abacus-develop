# Unified Sternheimer Basis-Optimization Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one ABACUS interface that preserves the validated molecular SIAB output and emits q-resolved, streamable periodic Delta-Sternheimer data for offline C-basis optimization.

**Architecture:** A canonical `out_sternheimer_basis_opt` Boolean dispatches to the existing molecular writer at Gamma and a new periodic writer for `sternheimer_q_index>0`. The periodic path constructs a complex full-Coulomb whitening transform for each q, projects source and response grids onto Bloch spherical-Bessel primitives at `k+q`, and writes checksummed binary chunks plus a manifest. All compilation and C++ test execution run on df or df_dcu; the laptop is used only for source edits and static checks.

**Tech Stack:** ABACUS C++17, MPI/OpenMP, LAPACK/BLAS, GoogleTest, SHA256 helpers, SIAB Python/PyTorch readers, Slurm on df/df_dcu.

---

## File Map

- `source/source_io/module_parameter/input_parameter.h`: canonical user switch.
- `source/source_io/module_parameter/read_input_item_output.cpp`: input parsing, alias, and validation.
- `source/source_io/test_serial/read_input_item_test.cpp`: switch and conflict tests.
- `source/source_lcao/module_ri/sternheimer_coulomb_whitening_complex.{h,cpp}`: q-dependent complex Hermitian whitening.
- `source/source_lcao/module_ri/test/sternheimer_coulomb_whitening_complex_test.cpp`: whitening unit tests.
- `source/source_lcao/module_ri/sternheimer_basis_opt_periodic.{h,cpp}`: periodic data records, binary chunks, manifest, and validation.
- `source/source_lcao/module_ri/test/sternheimer_basis_opt_periodic_test.cpp`: deterministic writer/reader and q-weight tests.
- `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`: periodic driver integration and Bloch primitive projection.
- `source/source_lcao/module_ri/CMakeLists.txt`: production sources.
- `source/source_lcao/module_ri/test/CMakeLists.txt`: test targets.

The matching reader and optimizer live in the separate `ABACUS-orbitals`
repository and receive their own implementation plan after this binary contract
is frozen by Task 3 tests. This plan does not mix commits across repositories.

### Task 1: Canonical basis-optimization switch

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_output.cpp`
- Modify: `source/source_io/test_serial/read_input_item_test.cpp`

- [ ] **Step 1: Write failing input tests**

Add tests that require:

```cpp
EXPECT_FALSE(param.input.out_sternheimer_basis_opt);
read_label("out_sternheimer_basis_opt", "true", param);
EXPECT_TRUE(param.input.out_sternheimer_basis_opt);
```

Also require the canonical switch to reject PW basis, non-Delta ST, missing
`bessel_nao_rcut`, and simultaneous `out_sternheimer_librpa`. Verify that the
legacy `out_sternheimer_siab=true` sets the same canonical Boolean.

- [ ] **Step 2: Run the server test and confirm RED**

Stage the commit to a clean df build tree and run:

```bash
ctest --test-dir build -R read_input_item_test --output-on-failure
```

Expected: compile failure because `out_sternheimer_basis_opt` does not exist.

- [ ] **Step 3: Implement the canonical switch and compatibility alias**

Add:

```cpp
bool out_sternheimer_basis_opt = false;
```

Register `out_sternheimer_basis_opt` as the canonical input. Keep
`out_sternheimer_siab` registered as a compatibility alias whose reader sets
the canonical Boolean. All runtime code reads the canonical Boolean.

- [ ] **Step 4: Re-run the server test and confirm GREEN**

Run the same `ctest` command. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/source_io/module_parameter/input_parameter.h \
        source/source_io/module_parameter/read_input_item_output.cpp \
        source/source_io/test_serial/read_input_item_test.cpp
git commit -m "feat(sternheimer): unify basis optimization output switch"
```

### Task 2: Complex periodic Coulomb whitening

**Files:**
- Create: `source/source_lcao/module_ri/sternheimer_coulomb_whitening_complex.h`
- Create: `source/source_lcao/module_ri/sternheimer_coulomb_whitening_complex.cpp`
- Create: `source/source_lcao/module_ri/test/sternheimer_coulomb_whitening_complex_test.cpp`
- Modify: `source/source_lcao/module_ri/CMakeLists.txt`
- Modify: `source/source_lcao/module_ri/test/CMakeLists.txt`

- [ ] **Step 1: Write a failing Hermitian whitening test**

Use a complex positive-semidefinite 3x3 fixture and require:

```cpp
const auto result = make_sternheimer_complex_coulomb_whitening(metric, 3, 1e-12);
EXPECT_EQ(result.retained_rank, 2);
EXPECT_LT(max_identity_error(metric, result.transform), 1e-12);
```

Add failure cases for materially non-Hermitian input, materially negative
eigenvalues, invalid dimensions, and invalid thresholds.

- [ ] **Step 2: Run the server target and confirm RED**

```bash
ctest --test-dir build -R sternheimer_coulomb_whitening_complex --output-on-failure
```

Expected: target or symbols are missing.

- [ ] **Step 3: Implement complex whitening**

Define:

```cpp
struct SternheimerComplexCoulombWhitening {
    int raw_dimension = 0;
    int retained_rank = 0;
    int discarded_rank = 0;
    double relative_threshold = 0.0;
    std::vector<double> eigenvalues;
    std::vector<std::complex<double>> transform;
    double max_orthonormality_error = 0.0;
};
```

Hermitize only after rejecting a material anti-Hermitian component. Diagonalize
with the repository LAPACK connector, reject eigenvalues below the negative
tolerance, retain `lambda > threshold*lambda_max`, and build
`W=U_r Lambda_r^-1/2`.

- [ ] **Step 4: Re-run RED target and related real-whitening regression**

```bash
ctest --test-dir build -R 'sternheimer_(coulomb_whitening|coulomb_whitening_complex)' --output-on-failure
```

Expected: both targets pass.

- [ ] **Step 5: Commit**

```bash
git add source/source_lcao/module_ri/sternheimer_coulomb_whitening_complex.* \
        source/source_lcao/module_ri/test/sternheimer_coulomb_whitening_complex_test.cpp \
        source/source_lcao/module_ri/CMakeLists.txt \
        source/source_lcao/module_ri/test/CMakeLists.txt
git commit -m "feat(sternheimer): whiten complex periodic Coulomb channels"
```

### Task 3: Versioned periodic binary contract

**Files:**
- Create: `source/source_lcao/module_ri/sternheimer_basis_opt_periodic.h`
- Create: `source/source_lcao/module_ri/sternheimer_basis_opt_periodic.cpp`
- Create: `source/source_lcao/module_ri/test/sternheimer_basis_opt_periodic_test.cpp`
- Modify: `source/source_lcao/module_ri/CMakeLists.txt`
- Modify: `source/source_lcao/module_ri/test/CMakeLists.txt`

- [ ] **Step 1: Write failing round-trip tests**

Construct one overlap, one source, and one response chunk with complex values.
Require exact metadata and numerical round-trip, deterministic file hashes,
atomic temporary-file rename, rejected truncated payloads, and rejected duplicate
`(iq,ik,ifreq)` records.

- [ ] **Step 2: Run the server target and confirm RED**

```bash
ctest --test-dir build -R sternheimer_basis_opt_periodic --output-on-failure
```

Expected: target or writer API is missing.

- [ ] **Step 3: Implement the chunk and manifest API**

Use a fixed little-endian header containing:

```cpp
struct PeriodicBasisOptChunkHeader {
    char magic[16];
    std::uint32_t version;
    std::uint32_t kind;
    std::int32_t iq;
    std::int32_t ik;
    std::int32_t ifrequency;
    std::uint64_t rows;
    std::uint64_t columns;
};
```

Payload values are complex128 stored as consecutive real/imaginary doubles.
The text manifest records q/k/frequency data, dimensions, physics hashes, chunk
paths, and SHA256 values.

- [ ] **Step 4: Re-run the round-trip test and confirm GREEN**

Run the same target. Expected: all cases pass with no leftover temporary files.

- [ ] **Step 5: Commit**

```bash
git add source/source_lcao/module_ri/sternheimer_basis_opt_periodic.* \
        source/source_lcao/module_ri/test/sternheimer_basis_opt_periodic_test.cpp \
        source/source_lcao/module_ri/CMakeLists.txt \
        source/source_lcao/module_ri/test/CMakeLists.txt
git commit -m "feat(sternheimer): define periodic basis optimization data"
```

### Task 4: Bloch primitive overlap and projection

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`
- Modify: `source/source_lcao/module_ri/test/test_sternheimer_siab_primitives.cpp`

- [ ] **Step 1: Write failing non-Gamma primitive tests**

Extend the existing fixture to use two non-Gamma k points. Require the
reciprocal primitive overlap to equal an independent FFT-grid overlap and
require `S(k)` to be Hermitian for both k points while differing between them.

- [ ] **Step 2: Run the server primitive target and confirm RED**

```bash
ctest --test-dir build -R sternheimer_siab_primitives --output-on-failure
```

Expected: the new periodic export helper is absent or uses Gamma primitives.

- [ ] **Step 3: Generalize the export helper**

Change the helper to accept `ik`, the full `PW_Basis_K`, and a destination
Bloch sector. Build primitives with
`Numerical_Basis::siab_primitive_reciprocal_values(ik, ...)`, compute the
complex `S(k)` from reciprocal coefficients, and project complete response
grids using the same k-sector FFT basis.

- [ ] **Step 4: Re-run serial and MPI primitive tests**

```bash
ctest --test-dir build -R sternheimer_siab_primitives --output-on-failure
```

Expected: serial and MPI2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_primitives.cpp
git commit -m "feat(sternheimer): project responses onto Bloch SIAB primitives"
```

### Task 5: Integrate periodic source and response output

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`
- Modify: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.h`
- Modify: `source/source_lcao/module_ri/test/sternheimer_abacus_st_smoke_test.cpp`

- [ ] **Step 1: Write failing dispatch and ownership tests**

Require the canonical switch plus positive q index to select periodic basis-opt
output, preserve the existing LibRPA path when the switch is false, assign each
source/response chunk to exactly one rank, and include q-star weights whose sum
over a campaign is one.

- [ ] **Step 2: Run the driver helper test and confirm RED**

```bash
ctest --test-dir build -R sternheimer_abacus_st_smoke --output-on-failure
```

Expected: periodic basis-opt dispatch is absent.

- [ ] **Step 3: Integrate q-specific data production**

In the existing periodic loop:

1. build `V(q)` from periodic ABFS densities and potentials;
2. whiten channels with the Task 2 transform;
3. build/cache `S(k+q)` primitives per destination k record;
4. project `v_tilde_h(q) psi[n,k]` into source chunks;
5. project each reconstructed Delta-ST response into response chunks;
6. reduce/gather only chunk ownership metadata, not dense all-q arrays;
7. write `status.dat` only after every expected chunk and checksum exists.

The normal periodic `chi0` accumulation remains unchanged when the new switch
is false.

- [ ] **Step 4: Run focused and existing periodic helper tests**

```bash
ctest --test-dir build -R 'sternheimer_(abacus_st_smoke|basis_opt_periodic|siab_primitives|coulomb_whitening)' --output-on-failure
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/source_lcao/module_ri/sternheimer_abacus_st_smoke.* \
        source/source_lcao/module_ri/test/sternheimer_abacus_st_smoke_test.cpp
git commit -m "feat(sternheimer): emit periodic basis optimization references"
```

### Task 6: Server build and molecular regression

**Files:**
- Create under project staging: a df build Slurm script and provenance manifest.

- [ ] **Step 1: Build the exact commit on df**

Use a new source/build/artifact directory. Record branch, commit, executable
SHA256, compiler/MPI/OpenMP dependencies, and build job ID. Do not overwrite the
atomic `56a2af8c` artifact.

- [ ] **Step 2: Run the focused C++ suites**

Require every Task 1-5 test to pass from the server build.

- [ ] **Step 3: Re-run the completed H/H2 molecular fixture**

Use `out_sternheimer_basis_opt=1` and require unchanged row counts,
frequencies, whitening rank/hash, projected-Pi matrices, and loss within the
existing text-output precision.

- [ ] **Step 4: Commit build/provenance records**

```bash
git add docs server_jobs
git commit -m "test(sternheimer): validate unified basis optimization output"
```

### Task 7: Diamond-C complete-q output production

**Files:**
- Create: canonical-q campaign inputs and Slurm scripts under `/work1` or the
  accepted df production root.
- Update: the AITP Delta-ST development TeX section and PDF.

- [ ] **Step 1: Freeze the production definition**

Record diamond C `a=3.57 Angstrom`, `45 Ry`, explicit `24^3`, FD8,
`k/q=4^3`, `nfreq=12`, explicit PCA=`1e-4` C ABFS, full Coulomb, and the exact
new executable hash.

- [ ] **Step 2: Submit one complete canonical-q campaign**

Use the known 13 q representatives and their q-star weights. Before submission,
reject active or completed duplicates with the same normalized input hash.

- [ ] **Step 3: Validate every q dataset**

Require scheduler success, `all_converged=yes`, finite residuals, complete
chunk manifests, common physics hashes, per-q whitening identity error below
`1e-8`, and q weights summing to one.

- [ ] **Step 4: Freeze the accepted output campaign**

Archive the common campaign manifest, q-level status and hashes, wall times,
node-hours, output sizes, and the exact reader contract for the follow-up
`ABACUS-orbitals` implementation plan.

- [ ] **Step 5: Merge the validated commits into `master_ghj`**

Preserve commit attribution, run the final regression suite from the merge
commit, push `master_ghj`, and record the remote branch head and binary hash.

After this plan completes, create and execute a separate
`ABACUS-orbitals` plan for the memory-mapped multi-q reader, Galerkin projected-
Pi loss, minibatch optimizer, complete-loss plateau, and matched SOS-RPA
promotion.
