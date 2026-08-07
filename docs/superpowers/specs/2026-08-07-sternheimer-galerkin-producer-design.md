# Fixed-AO Sternheimer Galerkin Producer Design

## Purpose

The first ABACUS gate for the finite-AO Galerkin Sternheimer route must isolate
the finite LCAO response from primitive-basis fitting.  It therefore exports
the actual converged LCAO Hamiltonian and overlap matrices together with the
same grid-sampled auxiliary Hartree perturbations used by Delta-ST.

This gate is not yet the differentiable primitive contract.  Primitive
matrices are added only after the fixed-TZDP result reproduces the all-bands
SOS response and LibRPA correlation energy.

## Fixed-AO Contract

For each physical collinear spin channel, ABACUS exports the converged
Gamma-point matrices in the exact LCAO ordering:

\[
  H^\sigma_{ab}=\langle\phi_a|\hat H^\sigma|\phi_b\rangle,
  \qquad
  S_{ab}=\langle\phi_a|\phi_b\rangle.
\]

ABACUS stores the LCAO Hamiltonian and eigenvalues in Ry.  The producer writes
both in Ha so that the matrix equation uses the same unit as the GreenX
imaginary frequencies.

For every generated PCA auxiliary channel, the producer evaluates

\[
  V^\mu_{ab}
  =\Delta\Omega\sum_h
    \phi_a^*(\mathbf r_h)\,v_\mu(\mathbf r_h)\,\phi_b(\mathbf r_h),
\]

where `v_mu` is the Ha-valued Hartree potential already used to form the
Delta-ST right-hand side.  The potential is spin independent, but the
Hamiltonian and occupations are spin resolved.

The file also records ABACUS reference eigenvalues and occupations.  The
consumer independently solves

\[
  H^\sigma C^\sigma=S C^\sigma\epsilon^\sigma
\]

and must reproduce the recorded eigenvalues before evaluating either
Galerkin Sternheimer or explicit SOS.

## Output Boundary

The existing `sternheimer_matrix.dat` SIAB-v1 file remains byte-compatible and
continues to contain the grid-reference projection target.  Fixed-AO matrices
are written to a separate versioned sidecar,
`sternheimer_galerkin_fixed_ao.dat`, with these sections:

1. `STERNHEIMER_GALERKIN_HEADER`: version, representation, dimensions and
   units.
2. `SPIN_METADATA`: spin index, state count, eigenvalues and occupations.
3. `OVERLAP_S`: one row-major complex Hermitian matrix.
4. `HAMILTONIAN_H`: one row-major complex Hermitian matrix per spin.
5. `PERTURBATION_V`: one row-major complex Hermitian matrix per auxiliary
   channel.
6. `FREQUENCY_GRID`: positive imaginary frequencies and weights in Ha.
7. `PROVENANCE_JSON`: the same source, executable, orbital, pseudopotential,
   auxiliary-basis, cell, cutoff, kernel and parallel metadata as SIAB-v1.

The sidecar is produced only with `out_sternheimer_siab=true`.  No new user
parameter is introduced in this first gate.

## Required Runtime Checks

- Gamma-point LCAO Delta-ST, `nspin=1` or `nspin=2`.
- `nbands == nlocal`; otherwise the all-bands finite-space comparison is not
  defined and output stops.
- Every distributed H/S element is gathered exactly once.
- S, every H, and every V are finite and Hermitian within `1e-10` absolute
  tolerance.
- S is positive definite in the consumer and its condition is reported.
- State/eigenvalue/occupation dimensions agree with `nlocal` and physical
  spin count.
- The auxiliary-channel count and ordering are identical to the Delta-ST and
  LibRPA writer path.
- Atomic replacement is used; a partial sidecar is never accepted.

## Validation Gates

### Gate A: writer and matrix algebra

Synthetic complex fixtures must verify canonical ordering, unit conversion,
Hermiticity rejection, dimensional rejection, deterministic output and
atomic replacement.  The PyTorch consumer must satisfy
`M_Galerkin == M_SOS` for the emitted fixed-AO matrices.

### Gate B: fixed H/H2 TZDP

Use the frozen H/TZDP chain with `exx_pca_thr=1e-4`.  Independently
diagonalize emitted H/S and compare all eigenvalues with ABACUS.  Then compare
the Galerkin and explicit all-bands SOS matrices built from emitted V, requiring
relative Frobenius error below `1e-10`.

### Gate C: ABACUS/LibRPA definition match

Compare the emitted-grid-V SOS response with the standard ABACUS-to-LibRPA SOS
response using the same full-Coulomb matrix, frequency grid, occupations and
auxiliary basis.  A discrepancy here is evidence about the perturbation
integral/kernel or normalization; it must not be hidden by basis optimization.

Only after Gate C passes is the analogous primitive contract
`S^p`, `H^{p,sigma}` and `V^{p,mu}` added and connected to trainable SIAB
coefficients.

