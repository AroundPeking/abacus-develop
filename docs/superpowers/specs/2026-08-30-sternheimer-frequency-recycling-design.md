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
operator to every new basis vector and form `W_j = A_j V`.  Determine the
reduced coefficients from the minimum-residual problem

`min_y ||b_j - W_j y||`.

The normal equations use `W_j^H W_j` and `W_j^H b_j`, matching the existing
GMRES least-squares convention.  Form provisional `x_j = V y_j`, then evaluate
the full-space residual `r_j = b_j - W_j y_j`.  If every relative residual
meets the existing tolerance, the group is accepted.  Otherwise select the
largest relative residual, apply that frequency's existing preconditioner,
project out `V`, normalize, and append the result.  This enrichment continues
until all frequencies converge or a dimension/iteration limit is reached.

This is recycled reduced-basis iteration, not multi-shift GMRES.  Every
frequency retains its own operator, RHS, preconditioner, residual, and stopping
decision.

The operator-family interface exposes the frequency-independent expensive
work explicitly.  For each new basis vector it projects the input and evaluates
the grid Hamiltonian once, then forms every `A_j V` column by adding that
frequency's shift and residual low-rank correction.  Independent per-frequency
operator callbacks remain available for deterministic fallback.  The result
records both conceptual per-frequency operator columns and the number of shared
family applications so a benchmark cannot mistake subspace reuse for actual
Hamiltonian reuse.  The Delta-ST result also counts independent fallback calls
and the final per-frequency physical-residual applications in its total
Hamiltonian-application field.

## First Scope

The first code layer is a general `FrequencyLinearProblem` solver in
`sternheimer_rpa`.  Unit tests use non-shifted, noncommuting frequency families
so an accidental multi-shift assumption cannot pass.  A separate family apply
test requires one callback per common basis vector and zero independent apply
calls before fallback.

The first Delta-ST integration is deliberately narrow:

- one occupied state and one auxiliary-channel batch at a time;
- a configurable group of low, middle, and high imaginary frequencies;
- frequencies resident in the same MPI group only;
- a hard common-basis dimension cap;
- explicit experimental enablement and provenance output.

The experiment is disabled unless
`ABACUS_STERNHEIMER_FREQUENCY_RECYCLING=true` is set.  The initial group size
and basis cap are controlled by
`ABACUS_STERNHEIMER_FREQUENCY_RECYCLING_GROUP_SIZE` and
`ABACUS_STERNHEIMER_FREQUENCY_RECYCLING_MAX_BASIS_DIMENSION`; their defaults
are 3 and 48.  Invalid or unsupported layouts are rejected rather than silently
running an unshared path.

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

The conservative estimator counts `(nfreq + 1) * basis_dimension + 4 * nfreq`
complex grid vectors: one common basis, all frequency-specific operator images,
and frequency RHS/solution/residual workspaces.  It is a gate estimate, not a
replacement for measured MaxRSS.

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

## Final Pilot Outcome

The df_dcu Si q=7 low/middle/high-frequency pilot completed all 256,512
equations with no fallback. The incremental projected-matrix update reduced
the recycling wall time from `1:08:16` to `37:13.28`; the two recycling
implementations agree within relative Frobenius `1.71194e-11`.

The optimized recycling path did not pass the independent-frequency gates:

- maximum response relative Frobenius difference: `2.12206e-10` (`1e-10`
  limit);
- Hamiltonian-application reduction: 12.81% (40% minimum);
- wall: `37:13.28` versus `33:32.86`, or `0.9013x` (1.5x minimum);
- node-hours: 4.9628 versus 4.4730;
- projected dot products: 248,723,874.

Therefore the full 12-frequency A/B was not submitted. The feature remains
opt-in on this branch and is not merged or enabled by default. The complete
physical and resource record is stored in
`server_jobs/sternheimer_performance_20260830/STERNHEIMER_FREQUENCY_RECYCLING_BENCHMARK_20260830.md`
in the calculation workspace.
