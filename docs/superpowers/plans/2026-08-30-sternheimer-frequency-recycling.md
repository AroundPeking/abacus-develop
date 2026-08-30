# Sternheimer Frequency-Recycling Implementation Plan

**Goal:** Test whether an adaptive common reduced space can reuse work across
Delta-ST imaginary frequencies while retaining frequency-specific equations
and full-residual acceptance.

**Architecture:** Add a generic frequency-family reduced solver to
`SternheimerRPA`, then integrate it behind an explicit runtime experiment for
small co-located frequency groups.  Any numerical or resource failure falls
back to the accepted independent-frequency GMRES path.

**Tech stack:** C++17, GoogleTest, CMake, MPI/OpenMP, FFTW, Slurm on df_dcu,
ABACUS Delta-Sternheimer, LibRPA.

---

### Task 1: Establish the exact baseline

**Files:** No source changes.

- [ ] Configure a clean Debug test build from current `origin/master_ghj`.
- [ ] Build and run `MODULE_RI_sternheimer_rpa_test`,
  `MODULE_RI_sternheimer_delta_test`, and
  `MODULE_RI_sternheimer_periodic_solver_test`.
- [ ] Record any environment-only limitation separately from source failures.

### Task 2: Add the generic adaptive frequency-family solver

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_rpa.h`
- Modify: `source/source_lcao/module_ri/sternheimer_rpa.cpp`
- Modify: `source/source_lcao/module_ri/test/sternheimer_rpa_test.cpp`

- [ ] Add failing tests for two noncommuting frequency-dependent operators,
  different RHS vectors, and different preconditioners.
- [ ] Add failing tests for empty RHS, dependent enrichment, dimension-cap
  fallback, and exact full-residual reporting.
- [ ] Introduce `FrequencyLinearProblem`, `FrequencyRecyclingOptions`, and a
  result containing per-frequency solver records, operator counts, basis size,
  and fallback reason.
- [ ] Implement orthonormal enrichment, explicit `V^H A_j V`, dense projected
  solves, and full-residual checks.
- [ ] Run the focused RPA test after each red-green-refactor step.

### Task 3: Build frequency-specific Delta-ST systems without changing physics

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_delta.h`
- Modify: `source/source_lcao/module_ri/sternheimer_delta.cpp`
- Modify: `source/source_lcao/module_ri/test/sternheimer_delta_test.cpp`

- [ ] Add a failing low/middle/high-frequency test comparing recycled and
  independent solves for frequency-dependent denominators, low-rank operator
  corrections, RHS vectors, and spectral preconditioners.
- [ ] Refactor one-frequency closure construction into a reusable helper whose
  scalar behavior is unchanged.
- [ ] Add an experimental small-frequency-group entry point and deterministic
  fallback to existing GMRES.
- [ ] Compare reconstructed response, coefficients, solver records, and full
  physical residuals within the design tolerances.

### Task 4: Add guarded runtime integration and resource accounting

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_runtime_options.h`
- Modify: `source/source_lcao/module_ri/sternheimer_runtime_options.cpp`
- Modify: `source/source_lcao/module_ri/sternheimer_channel_resources.h`
- Modify: `source/source_lcao/module_ri/sternheimer_channel_resources.cpp`
- Modify: periodic producer/smoke files selected after source tracing.
- Modify: matching runtime/resource/smoke tests.

- [ ] Add failing parser tests for disabled default, explicit enablement,
  frequency-group size, basis cap, and invalid values.
- [ ] Reject layouts that would require cross-group basis communication in the
  first implementation.
- [ ] Add basis/operator-image memory accounting to the existing worker budget.
- [ ] Emit effective settings, operator counts, basis size, fallback state, and
  reason in provenance.
- [ ] Preserve output ordering and the current independent-frequency default.

### Task 5: Complete local and remote regression

**Files:** Modify only when tests expose defects.

- [ ] Build and run all affected RPA, Delta-ST, preconditioner, runtime,
  resource, periodic, and smoke tests.
- [ ] Run `git diff --check origin/master_ghj...HEAD` and audit that no physical
  defaults, tolerances, frequency grids, FD order, or output definitions moved.
- [ ] Build one immutable RelWithDebInfo artifact on df_dcu `/work1` and require
  the complete affected test set to pass.

### Task 6: Run the Si three-frequency physical A/B

**Files:**
- Create: benchmark submission and comparison scripts under
  `server_jobs/sternheimer_performance_20260830/`.
- Create: `STERNHEIMER_FREQUENCY_RECYCLING_BENCHMARK_20260830.md`.

- [ ] Reuse the accepted Si q case and select actual low, middle, and high
  GreenX frequencies; do not run any physics locally.
- [ ] Run independent and recycled paths with identical executable, inputs,
  MPI/OMP resources, channel batch width, and convergence tolerance.
- [ ] Require complete physical output, full residuals, response-matrix gate,
  Hamiltonian-application count, wall time, MaxRSS, and node-hours.
- [ ] Stop without production integration unless both numerical and performance
  gates in the design are met.

### Task 7: Decide whether to continue or stop

**Files:**
- Modify benchmark/development documentation and production skill only after a
  verified physical result.

- [ ] If the pilot fails, record the cause and keep the feature branch opt-in.
- [ ] If the pilot passes, design the MPI-local grouping needed for a full Si
  workload and run a complete energy/resource A/B.
- [ ] Merge and consider default enablement only after the complete workload is
  result-preserving and faster end to end.
