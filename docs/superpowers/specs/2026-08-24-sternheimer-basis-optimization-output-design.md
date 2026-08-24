# Unified Sternheimer Basis-Optimization Output Design

## Objective

Add one production interface to ABACUS `master_ghj` that emits immutable
Delta-Sternheimer reference data for atomic, molecular, and periodic SIAB basis
optimization.  The expensive Sternheimer equations are solved once for a fixed
physical definition.  Candidate atomic orbitals are then optimized offline and
validated by a matched SOS-RPA calculation.

The user-facing switch is:

```text
out_sternheimer_basis_opt 1
```

The existing `out_sternheimer_siab` input remains a compatibility alias.  If
both labels are present they must have the same value.  Internally there is one
canonical Boolean, so the two labels cannot select different implementations.

## Scope

The first production version supports:

- isolated atoms and molecules at Gamma;
- non-spin-polarized insulating periodic solids with a positive PBE gap;
- Delta-Sternheimer with explicit FD8;
- explicit `ABFS_ORBITAL` provenance;
- full-Coulomb auxiliary perturbations;
- complex Bloch sectors from `k` to `k+q`;
- multiple GreenX imaginary frequencies;
- canonical-q calculations with q-star weights;
- offline Galerkin/projected-Pi optimization with a fixed DFT safeguard.

The analytic q-average head and wing are not training targets in the first
version.  They remain enabled in the final matched SOS-RPA validation.  The
training target is the finite-q/body response represented by the solved
first-order wavefunctions.

## Execution Modes

`out_sternheimer_basis_opt=1` selects the path from existing calculation data:

1. If `sternheimer_q_index=0`, ABACUS writes the molecular/atomic basis-
   optimization dataset.  Existing validated molecular SIAB v1 output remains
   readable.
2. If `sternheimer_q_index>0`, ABACUS writes one periodic dataset for that
   canonical q representative.
3. The full solid reference is one campaign containing every canonical q.  It
   may consist of multiple scheduler jobs, but all jobs share one executable,
   PP, NAO, ABFS, frequency grid, FD order, real-space grid, and k/q mesh.

The switch remains mutually exclusive with `out_sternheimer_librpa` in the
first version.  This avoids silently mixing the raw LibRPA atom-block auxiliary
representation with the Coulomb-whitened optimization representation.

## Periodic Physics Contract

Let `B` contain radial coefficients that map atom-centered spherical-Bessel
primitives to candidate numerical atomic orbitals.  For a response from
`k` to `k+q`, define the candidate overlap

```text
G_B(k+q) = B^dagger S(k+q) B,
```

where `S(k+q)` is the complex Bloch primitive overlap.  For occupied state
`n,k`, Coulomb-whitened auxiliary perturbation `h`, and imaginary frequency
`omega`, ABACUS exports:

```text
D[n,k,q,h]       = <primitive(k+q) | v_tilde_h(q) psi[n,k]>
Q[n,k,q,h,omega] = <primitive(k+q) | delta psi[n,k,h](q,omega)>.
```

The candidate Galerkin response is

```text
A_B(q,omega) = sum[n,k] f[n,k]
               D[n,k,q] B G_B(k+q)^-1 B^dagger Q[n,k,q,omega]^dagger,

Pi_B(q,omega) = A_B(q,omega) + A_B(q,omega)^dagger.
```

The reference `Pi_ref` is evaluated in the retained primitive span.  Different
q points have different complex auxiliary gauges and are never combined as
matrices.  Only gauge-invariant scalar losses are combined:

```text
L_Pi(B) =
  sum[q,j] w_q w_j ||Pi_B(q,omega_j)-Pi_ref(q,omega_j)||_F^2
  ----------------------------------------------------------------,
  sum[q,j] w_q w_j ||Pi_ref(q,omega_j)||_F^2

L_total(B) = L_Pi(B) + lambda_DFT L_DFT(B).
```

Here `w_q` is the q-star multiplicity divided by the full q-mesh size and
`w_j` is the GreenX quadrature weight.  The DFT term keeps the occupied/low-
energy NAO description fixed or penalizes its degradation.

## Coulomb Gauge

For every q, construct the complex Hermitian full-Coulomb metric

```text
V(q) = U(q) Lambda(q) U(q)^dagger.
```

Retain eigenvalues above the explicit relative threshold and define

```text
W(q) = U_r(q) Lambda_r(q)^(-1/2),
W(q)^dagger V(q) W(q) = I.
```

The exported perturbations use `W(q)`.  Each q dataset records the complete
eigenvalue spectrum, retained rank, threshold, orthonormality error, and
transform hash.  A response and source pair is accepted only when all hashes
and dimensions match within the same q dataset.

## File Contract

Do not extend the molecular text v1 format with ambiguous periodic fields.
Periodic output uses a new versioned directory:

```text
OUT.ABACUS/STERNHEIMER_BASIS_OPT_V1/
  manifest.dat
  coulomb_whitening.dat
  primitive_blocks.dat
  overlap_ik_<ik>.bin
  source_ik_<ik>.bin
  response_ik_<ik>_ifreq_<ifreq>.bin
  status.dat
```

`manifest.dat` contains the q index/vector/weight, k mesh, selected k records,
frequency values and weights, FD order, actual real-space grid, PP/NAO/ABFS
hashes, executable commit/hash, primitive definition, complex dtype, and every
binary chunk hash.  Binary chunks have a magic string, version, dimensions,
indices, and payload checksum.  Files are written through temporary paths and
renamed only after complete writes.

Text output is deliberately avoided for periodic response arrays because it
would create impractically large files and slow parsing.  The optimizer must
memory-map or stream the binary chunks and must not load the full all-q dataset
at once.

## Offline Optimization

The SIAB reader accepts a campaign manifest listing all canonical-q datasets.
It validates common physical provenance and q-star weight normalization before
constructing the loss.

Training uses stratified q/k/frequency/channel minibatches with unbiased
physical weights.  A complete streamed loss is evaluated at fixed checkpoints
and for final promotion.  The exact full-dataset result, not the minibatch
estimate, determines whether a basis is accepted.

An optional deterministic low-rank preprocessing stage may compress response
rows after the uncompressed reader is validated.  Compression is accepted only
if it changes the complete projected-Pi loss by less than `1e-6` relative and
the reconstructed RPA correlation contribution by less than
`0.01 kcal/mol/atom` on the reference basis.

Expected performance is assessed rather than assumed.  Record:

- one minibatch loss-plus-gradient wall time;
- one complete all-q loss wall time;
- peak resident memory;
- preprocessing wall time and compression ratio;
- Delta-ST reference node-hours.

The optimization is successful only if it is materially cheaper than solving
Delta-ST again and produces a finite, reproducible gradient.

## Diamond-C Production Definition

The first periodic production target is diamond C with:

```text
lattice constant       3.57 Angstrom
ecutwfc                45 Ry
actual FD grid          24 x 24 x 24
sternheimer_fd_order    8
k mesh                  4 x 4 x 4
q mesh                  4 x 4 x 4
sternheimer_nfreq       12
C auxiliary basis       explicit PCA=1e-4 ABFS
Coulomb kernel          full periodic Coulomb
sqrt-Coulomb threshold  1e-5 for matched RPA validation
```

The `24^3` grid is selected because the completed `24^3` to `30^3` FD8
correlation-energy change is `0.0768649 kcal/mol/cell`, or
`0.0384324 kcal/mol/C`.  The old `30^3` point is the independent confirmation,
not the production training grid.  The explicit PCA=`1e-4` ABFS is a deliberate
change from the old PCA=`1e-6` grid-convergence campaign and is recorded as a
new auxiliary-space definition.

## Integration Strategy

Development starts from the current source-v1 feature commit, which is already
a descendant of periodic Delta-ST and the FD8 default.  Each development build
is isolated by commit and executable hash.  The existing atomic executable is
never overwritten.

After unit tests, molecular regression, periodic source/response validation,
and one successful complete diamond-C campaign, the commits are merged into
`master_ghj`.  The final supported state is one `master_ghj` branch and one
user-facing basis-optimization switch for molecules and solids; the temporary
development branch is not a second production line.

## Validation Gates

1. Input tests prove the new switch, compatibility alias, conflict checks, and
   molecular/periodic dispatch.
2. Complex whitening tests require Hermiticity and
   `||W^dagger V W-I||_max <= 1e-8` after truncation.
3. A periodic synthetic test verifies `k -> k+q`, complex primitive overlap,
   q metadata, chunk checksums, and q-star weights.
4. The existing H/H2 molecular projected-Pi result is unchanged within current
   parser/output precision.
5. Candidate `B=I` reproduces the retained-primitive `Pi_ref` for every tested
   q and frequency.
6. q-star weighted assembly agrees with explicit full-q assembly on a small
   deterministic fixture.
7. MPI layouts produce identical manifests and matrices within solver/output
   precision.
8. The complete diamond-C output reports `status success`,
   `all_converged=yes`, complete chunks, finite residuals, matching hashes, and
   normalized q weights.
9. The promoted optimized basis passes a matched full-Coulomb SOS-RPA
   comparison with analytic q-average head/wing enabled.

No production job is submitted before the implementation passes gates 1-7.
