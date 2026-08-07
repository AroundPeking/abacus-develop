# Fixed-AO Galerkin Fast Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent ABACUS mode that writes the fixed-LCAO Galerkin sidecar before any Sternheimer linear equation is solved.

**Architecture:** A new input flag selects fixed-AO export independently of LibRPA chi0 and SIAB first-order targets.  The LCAO controller gathers H/S for either fixed-output route, while the producer extracts one shared sidecar-writing helper and returns early only in fixed-only mode.  Existing SIAB and LibRPA paths remain behaviorally unchanged.

**Tech Stack:** C++17, ABACUS input registry, Gamma-point LCAO controller, MPI, GoogleTest, Slurm on df_dcu.

---

### Task 1: Register the independent input

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_output.cpp`
- Test: `source/source_lcao/module_ri/test/test_sternheimer_siab_writer.cpp`

- [ ] **Step 1: Write a failing registration test**

Add `SternheimerGalerkinInput.RegisteredCheckAllowsIndependentFixedAOOutput`.
The test must find `out_sternheimer_galerkin`, assert its default is false,
accept `basis_type=lcao`, `sternheimer_delta=true`, and empty
`bessel_nao_rcuts` while `out_sternheimer_librpa=false`, and reject PW or
`sternheimer_delta=false`.

- [ ] **Step 2: Run the registration test on df_dcu and verify RED**

Run the existing `MODULE_RI_sternheimer_siab_writer_test` target with a test
filter for `SternheimerGalerkinInput.*`.  Expected failure: the input item is
absent.

- [ ] **Step 3: Add the input field and registry item**

Add:

```cpp
bool out_sternheimer_galerkin = false;
```

Register `out_sternheimer_galerkin` as a Boolean output item.  Its check must
require `basis_type=lcao` and `sternheimer_delta=true`; it must not inspect
`out_sternheimer_librpa` or `bessel_nao_rcuts`.

- [ ] **Step 4: Run writer/input tests and verify GREEN**

Expected: the new filtered test and all existing writer/input tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat(sternheimer): register fixed AO Galerkin output`.

### Task 2: Specify controller mode decisions

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.h`
- Test: `source/source_lcao/module_ri/test/sternheimer_abacus_st_smoke_test.cpp`

- [ ] **Step 1: Write failing pure mode-decision tests**

Specify helpers that return true when the producer must run, when fixed H/S
must be gathered, and when fixed-only output must return early.  Cover the four
input combinations in the design table.

- [ ] **Step 2: Run the smoke test on df_dcu and verify RED**

Expected failure: the mode-decision helpers do not exist.

- [ ] **Step 3: Implement the minimal Boolean helpers**

The decisions are:

```text
run = out_sternheimer_librpa || out_sternheimer_galerkin
fixed = out_sternheimer_siab || out_sternheimer_galerkin
fixed_only = out_sternheimer_galerkin && !out_sternheimer_librpa
```

- [ ] **Step 4: Run the smoke tests and verify GREEN**

Expected: every combination passes without touching production state.

- [ ] **Step 5: Commit**

Commit message: `test(sternheimer): specify fixed AO output modes`.

### Task 3: Wire the controller and extract early sidecar writing

**Files:**
- Modify: `source/source_io/module_ctrl/ctrl_scf_lcao.cpp`
- Modify: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`
- Test: `source/source_lcao/module_ri/test/sternheimer_abacus_st_smoke_test.cpp`

- [ ] **Step 1: Add a failing early-output trace test**

Extend the smoke seam with a synthetic trace that records sidecar writes and
linear-equation starts.  Fixed-only mode must report one write and zero starts;
full SIAB mode must report one write and a nonzero solver path.

- [ ] **Step 2: Run the focused smoke test on df_dcu and verify RED**

Expected failure: fixed-only mode neither enters the producer nor exposes an
early write boundary.

- [ ] **Step 3: Update the LCAO controller**

Enter the producer for either output flag.  Gather fixed H/S when
`out_sternheimer_siab || out_sternheimer_galerkin`.  Retain the existing PW
FFT-basis requirement only for `out_sternheimer_siab`, because primitive
targets alone need it.

- [ ] **Step 4: Extract and call the fixed-AO writer helper**

Move auxiliary metadata, AO value extraction, `build_fixed_ao_data`,
provenance, and `write_fixed_ao_v1` into one internal helper.  Call it after
frequency grid, full grid, channels, potentials, and sampled AO functions are
ready.  Use a complete grid whenever fixed output is active.

- [ ] **Step 5: Add the fixed-only return**

After rank 0 writes the sidecar, return on all ranks when
`out_sternheimer_galerkin && !out_sternheimer_librpa`.  Do this before solver
options, response buffers, primitive construction, or any occupied-state loop.

- [ ] **Step 6: Run smoke, writer, overlap, and fixed-AO tests**

Expected: all focused suites pass and the legacy sidecar fixture remains
byte-identical.

- [ ] **Step 7: Commit**

Commit message: `feat(sternheimer): export fixed AO matrices before solves`.

### Task 4: Formal remote verification and documentation

**Files:**
- Modify: `docs/superpowers/evidence/2026-08-07-fixed-ao-green.slurm`
- Modify: `docs/superpowers/evidence/2026-08-07-fixed-ao-submission.txt`
- Modify outside repository: `sternheimer_siab_project/main.tex`

- [ ] **Step 1: Archive an exact feature-branch commit**

Create a tracked-source archive, record SHA256, branch, commit, and controller
hash, and transfer it through the active df_dcu ControlMaster.

- [ ] **Step 2: Submit only to normal with full node resources**

Request one node, 30 CPUs, 110610 MiB, and 24 hours.  Build a fresh
LibRI/LibComm/DEBUG_INFO `abacus_3p`; run focused tests before the full build.

- [ ] **Step 3: Verify the formal result**

Require zero exit status, all focused tests green, an up-to-date executable,
and `ENABLE_LIBRI=ON`, `ENABLE_LIBCOMM=ON`, `DEBUG_INFO=ON` in CMakeCache.

- [ ] **Step 4: Run a one-rank H fixed-AO smoke**

Use `out_sternheimer_galerkin=1`, `nbands=nlocal`, 16 fixed GreenX
frequencies, `exx_pca_threshold=1e-4`, and no explicit `ABFS_ORBITAL`.  Require
`sternheimer_galerkin_fixed_ao.dat` and verify that no chi0 or first-order
target file is produced.

- [ ] **Step 5: Analyze the sidecar**

Run the committed SIAB reader and require generalized-eigenvalue agreement,
Galerkin/SOS relative matrix error below `1e-10`, and zero solved equations in
the ABACUS status output.

- [ ] **Step 6: Update TeX and commit evidence**

Record formulas, source and executable hashes, job resources, wall time,
matrix errors, and the fact that no linear equation was solved.
