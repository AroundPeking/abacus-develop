# Fixed-AO Sternheimer Galerkin Producer Implementation Plan

## Phase 1: Freeze the sidecar format with tests

- Add a writer fixture test for `sternheimer_galerkin_fixed_ao.dat`.
- Add failure tests for bad dimensions, non-Hermitian/nonfinite S/H/V,
  incomplete spin metadata and nonpositive frequencies.
- Run the new tests on df_dcu before implementation and retain the expected
  RED evidence.

## Phase 2: Implement the standalone writer

- Add fixed-AO data structures and a dedicated v1 writer without changing the
  existing SIAB-v1 writer.
- Reuse exact double formatting, provenance validation and atomic replacement
  through a shared internal utility only where this does not alter the
  canonical SIAB-v1 fixture.
- Rerun writer, provenance and SIAB-v1 regressions on df_dcu.

## Phase 3: Export converged LCAO H/S

- Gather Gamma-point H/S from `HamiltLCAO` for every physical spin channel.
- Convert H and reference eigenvalues from Ry to Ha at the output boundary.
- Require `nbands == nlocal` and record all occupations.
- Add serial and MPI matrix-gather unit tests before wiring the production
  call.

## Phase 4: Build auxiliary perturbation matrices

- Reuse the already sampled LCAO basis functions and Ha-valued auxiliary
  Hartree potentials.
- Compute `V_mu[a,b]` with the same complete uniform grid and
  `DeltaOmega` used by Delta-ST.
- Compute only the upper triangle and fill the lower triangle by conjugation.
- Add analytic grid tests for complex basis functions, Hermiticity, channel
  ordering and the volume element.

## Phase 5: Production wiring and remote regression

- Write the sidecar from rank 0 after all fixed matrices and metadata are
  complete.
- Build a fresh ABACUS executable on df_dcu and run only on the `normal`
  partition for formal tests.
- First use a small fixed-TZDP H case; then run the frozen H/H2 case with all
  bands and `exx_pca_thr=1e-4`.
- Feed the emitted matrices to the finite-AO Galerkin/SOS consumer and compare
  eigenvalues, response matrices and full-Coulomb LibRPA correlation energies.
- Record every result, exact source/executable/input hash and job resource in
  the research TeX document before starting primitive-matrix work.

