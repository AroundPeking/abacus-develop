# Fixed-AO Galerkin Fast Output Design

## Purpose

The fixed-AO Galerkin/SOS gate needs only the converged LCAO `H` and `S`, the
auxiliary Hartree perturbation matrices `V`, and the selected imaginary
frequency grid.  It does not need any first-order wavefunction or chi0 solve.
The current producer writes `sternheimer_galerkin_fixed_ao.dat` only after all
Delta-Sternheimer equations finish, which makes the algebra gate as expensive
as the grid reference calculation.

Add an independent output mode that writes the fixed-AO sidecar immediately
after the common matrix inputs are available and then returns before solver
allocation or any Sternheimer equation.

## Input Contract

Add the Boolean input `out_sternheimer_galerkin`, default `false`.

When true, it requires:

- `basis_type=lcao`;
- Gamma-point `nspin=1` or `nspin=2`;
- `sternheimer_delta=true`, selecting the existing LCAO zero-order path;
- `nbands=nlocal`, so the emitted finite AO space has all SOS virtual states.

It is independent of `rpa`, `out_sternheimer_librpa`, and
`out_sternheimer_siab`.  It does not require `bessel_nao_rcut`, because no
spherical-Bessel primitive or first-order target is constructed.

The four supported combinations are:

| Inputs | Behavior |
|---|---|
| `out_sternheimer_galerkin=1`, other Sternheimer outputs off | Write only the fixed-AO sidecar and return before all linear solves. |
| `out_sternheimer_siab=1` | Preserve the existing full Delta-ST target and chi0 path and also write the fixed-AO sidecar once. |
| both fixed-AO and SIAB outputs on | Run the full SIAB path and write one fixed-AO sidecar; do not return early. |
| `out_sternheimer_librpa=1` only | Preserve the existing chi0 behavior exactly. |

## Data Flow

The LCAO controller enters the Sternheimer producer when either
`out_sternheimer_librpa` or `out_sternheimer_galerkin` is true.  It gathers the
spin-resolved converged H/S matrices when either fixed-AO output route is
active.  The producer then builds, in order:

1. the GreenX or user-supplied frequency grid;
2. the product-PCA auxiliary density channels and their Ha Hartree potentials;
3. the complete uniform real-space grid;
4. the sampled LCAO functions;
5. `V_mu[a,b] = DeltaOmega sum_h phi_a*(h) v_mu(h) phi_b(h)`;
6. `sternheimer_galerkin_fixed_ao.dat` and its provenance.

Fixed-AO output always selects the complete real-space grid.  In a serial run
this is the ordinary local grid.  In an MPI run it uses the same all-gathered
full grid definition as frequency-MPI, preventing a partial z-slab integral.

If only `out_sternheimer_galerkin` is active, all ranks return after the
sidecar has been written by rank 0.  Solver options, chi0 response buffers,
Delta-ST projectors, primitive targets, and linear solves are never created.

## Compatibility

- Keep `out_sternheimer_siab` validation and output unchanged.
- Keep `sternheimer_matrix.dat` byte-compatible.
- Keep existing LibRPA chi0 files and status files unchanged.
- Keep the fixed-AO v1 sidecar format unchanged.
- Do not introduce a new kernel or normalization; `V` uses the same generated
  Ha Hartree potentials and uniform-grid volume element as the existing path.

## Tests

1. Input registration: default false; accept LCAO Delta mode without
   `out_sternheimer_librpa` or `bessel_nao_rcut`; reject PW and non-Delta use.
2. Mode decision: fixed-only mode enters the producer and requests H/S, while
   legacy LibRPA-only mode does not request fixed matrices.
3. Early-output helper: a synthetic fixed-AO fixture writes the sidecar and
   reports `solved_equations=0` without constructing solver state.
4. Compatibility: the existing SIAB input tests, writer fixture, grid
   integration tests, and full ABACUS build remain green.

The first production run uses one MPI rank and one full 30-thread normal node.
It records source, executable, orbital, pseudopotential, input, frequency, and
effective auxiliary-basis hashes before any H/H2 comparison.

## Non-goals

- No primitive `S^p/H^p/V^p` output in this change.
- No orbital optimization in this change.
- No change to Delta-ST or standard ST solvers.
- No claim of AO completeness from Galerkin/SOS equality.
