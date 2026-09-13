# Frozen-Density Periodic jY Operator Export

**Goal:** Complete finite-q high-angular-momentum matrices for the approved C jY ladder without solving another Delta-ST response or running a new SCF cycle.

**Architecture:** Extend the existing `sternheimer_siab_source_only` opt-in to the primitive-cell periodic path. Reuse the existing grid, Coulomb whitening, occupied sampling, and primitive S/H/O/D assembly. Write an explicitly distinct operators-only manifest; never write reference-response or response chunks, or claim equation convergence. Existing full-response and molecular defaults remain unchanged.

**Execution:** DF only; no local native compilation. Use the frozen 711af860c source as the baseline and preserve it. An isolated NSCF run must load the accepted binary charge density and reproduce the accepted Gamma matrices after the documented gauge alignment before any finite-q export is admitted.

## Tasks

- [x] Audit restart availability and the existing periodic rejection of source-only mode.
- [ ] Add remote-tested manifest contracts: distinct format, complete per-k operators, no response chunks, immutable charge-density provenance.
- [ ] Extend the periodic driver: require NSCF/file density; skip Delta subspace, response equations and response outputs; skip the unrelated RPA reader producer for this mode.
- [ ] Run the existing and new tests remotely; verify source-only routing and unchanged default format.
- [ ] Build one isolated diagnostic executable using the DF frozen toolchain; record source, dependency, cache and executable hashes.
- [ ] Run one Gamma frozen-density compatibility gate, then one finite-q known-spd gate. Reject any mismatch instead of treating restart existence as Hamiltonian equivalence.
- [ ] Only after compatibility passes, fill the remaining seven q-star operator sets and evaluate the spd/spdf/spdfg ladder against the existing 12-frequency reference.
- [ ] Only after full-q accuracy is established, compress shared atom-centred radial functions and measure the induced RPA error.

## Acceptance Boundaries

- `operators_only` is not a response dataset, an SOS calculation, or physical acceptance.
- A successful read of density does not prove the final SCF Hamiltonian is identical. Known S/H/O/D and occupied-energy comparisons are mandatory.
- The source-only manifest includes the loaded charge file hash, exact frequency metadata, q/k weights, original PP/orbital hashes, and every operator chunk hash.
- No zero-padding of missing high-l blocks, inference of missing finite-q response from Gamma, or extrapolation of Gamma improvement to full q.
- Physical target remains the existing full-q body RPA reference; head/wing and GW are separate later checks.
