# SIAB-Compatible Stack

This branch pins the `ABACUS` side of the currently validated SIAB workflow for CP2K GTH based
SOC calculations.

## Pinned repositories

- `ABACUS`: `AroundPeking/abacus-develop` branch `siab-v3907-baseline`
- `SIAB`: `AroundPeking/ABACUS-orbitals` branch `siab-abacus-v3907-compat`

## Code baseline

- `ABACUS` code base commit: `2d87d673e4d0c4edcd316abc0e04dc9988b4fe59`
- This branch keeps the code at that baseline and adds only this pinning note.

## Why this stack exists

- The validated `WSe2` workflow should not depend on a server-local legacy binary.
- Current SIAB still expects the old `istate.info` path, while newer `ABACUS` writes `eig.txt`.
- The paired `SIAB` branch adds a reader fallback from `istate.info` to sibling `eig.txt`.
- The paired `SIAB` branch already writes `bessel_nao_rcut` into generated `INPUT`, so the same
  SIAB tree can be reused on different servers.

## How to use

Clone and build `ABACUS` from this branch, then clone `ABACUS-orbitals` from
`siab-abacus-v3907-compat` and use that SIAB tree for orbital generation. Do not rely on an
untracked server-local `ABACUS.mpi` binary when reproducing this workflow on a new machine.
