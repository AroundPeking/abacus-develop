# Sternheimer Frequency-Recycling Design

## Objective

Reduce the repeated Hamiltonian work across imaginary frequencies in periodic
Delta-Sternheimer calculations without assuming a shifted linear-system form
that the production equations do not satisfy.  The first implementation is an
opt-in experiment.  The accepted independent-frequency GMRES path remains the
default and rollback route until a same-input Si calculation passes numerical,
residual, and wall-time gates.

## Why Standard Multi-Shift Krylov Is Not Applicable

For frequency index `j`, the production equation is a family

`A_j x_j = b_j`,

not `(A + sigma_j I) x_j = b`.  The Delta-ST denominators change the right-hand
side and low-rank correction with frequency, while the FD spectral
preconditioner also contains the frequency.  Restarted flexible GMRES further
breaks the residual-collinearity property required by standard multi-shift
methods.  Reusing one Arnoldi recurrence as if only a scalar shift changed
would therefore be mathematically unjustified.

## Selected Algorithm

Use an adaptive common reduced space for a small frequency group.  Let `V` be
an orthonormal grid-vector basis.  For each frequency, explicitly apply its own
operator to every new basis vector and form

`Abar_j = V^H A_j V`, `bbar_j = V^H b_j`.

Solve the small dense systems for provisional `x_j = V y_j`, then evaluate the
full-space residual `r_j = b_j - A_j x_j`.  If every relative residual meets
the existing tolerance, the group is accepted.  Otherwise select the largest
relative residual, apply that frequency's existing preconditioner, project out
`V`, normalize, and append the result.  This enrichment continues until all
frequencies converge or a dimension/iteration limit is reached.

This is recycled reduced-basis iteration, not multi-shift GMRES.  Every
frequency retains its own operator, RHS, preconditioner, residual, and stopping
decision.

## First Scope

The first code layer is a general `FrequencyLinearProblem` solver in
`sternheimer_rpa`.  Unit tests use non-shifted, noncommuting frequency families
so an accidental multi-shift assumption cannot pass.

The first Delta-ST integration is deliberately narrow:

- one occupied state and one auxiliary-channel batch at a time;
- a configurable group of low, middle, and high imaginary frequencies;
- frequencies resident in the same MPI group only;
- a hard common-basis dimension cap;
- explicit experimental enablement and provenance output.

The existing frequency-MPI decomposition assigns frequencies to different
ranks.  The prototype must not add all-to-all basis traffic.  A physical pilot
therefore co-locates only the selected three frequencies for measurement.  A
production grouping policy is designed only if the pilot shows enough benefit.

## Memory and Cost Controls

The common basis stores approximately `16 * Ngrid * basis_dimension` bytes per
vector family before operator images and solver workspaces.  The prototype
defaults to a conservative dimension cap of 48 and refuses activation when
the rank memory estimate exceeds the existing channel-worker budget.  It does
not multiply the outer channel-worker count by the frequency-group size.

Operator applications are counted separately for every frequency.  Reduced
dense solves are not treated as the bottleneck.  The pilot must report total
Hamiltonian applications, response-stage wall time, rank MaxRSS, and node-hours.

## Failure and Fallback Behavior

An empty RHS converges immediately.  Linear dependence during enrichment,
dense projected-system failure, a dimension-cap hit, or any full-residual
failure causes a deterministic fallback to the existing independent-frequency
GMRES for the affected group.  Fallback is recorded and never presented as an
accelerated result.  No response matrix is accepted from projected residuals
alone.

## Numerical Gates

Development follows failing-test-first cycles.  Local acceptance requires:

- exact input validation and deterministic fallback tests;
- independent reference solutions reproduced within `1e-12` in unit systems;
- full relative residuals no larger than the existing solver tolerance;
- a noncommuting frequency-family test, including frequency-dependent RHS and
  preconditioners;
- all directly affected RPA, Delta-ST, runtime-option, and periodic tests pass.

The remote Si low/middle/high-frequency A/B uses one immutable executable and
identical physical inputs.  It must satisfy:

- `all_converged=yes` and complete response/provenance artifacts;
- response-matrix relative Frobenius difference no larger than `1e-10`;
- no physical-residual regression;
- at least 40% fewer Hamiltonian applications;
- at least `1.5x` response-stage wall-time speedup without unacceptable MaxRSS.

Failure of either performance gate ends the experiment without changing the
production default.  Passing the three-frequency pilot permits, but does not
itself authorize, a full frequency-group production rollout.

## Integration Decision

The feature branch is `codex/sternheimer-frequency-recycling-20260830` from
the current `origin/master_ghj`.  The experiment stays opt-in until the full Si
energy and resource A/B also passes.  Only a result-preserving, measured
end-to-end improvement may be merged and later enabled by default.
