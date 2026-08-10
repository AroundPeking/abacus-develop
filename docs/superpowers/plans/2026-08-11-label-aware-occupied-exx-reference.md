# Label-Aware Occupied EXX Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a finite-BvK dense reference that reproduces all four LibRI EXX label paths and proves the exact AO--occupied factorization for the complete H matrix before any GaAs k444 EXX submission.

**Architecture:** Expand the serial C/V/D snapshots into a small finite-supercell tensor. Each stored local-RI C block contributes in both AO orders, so the resulting pair coefficient contains both possible auxiliary anchors and its dense CVCD expansion contains the four LibRI labels. This is the source-defined normalization: ABACUS stores the same-atom/same-cell C block with an explicit factor `0.5`, while the two AO-order additions restore the complete coefficient. Factor the assembled positive density globally, contract only its connected AO leg, convert the resulting H back to LibRI translation blocks, and compare both dense and occupied results with the production H snapshot. The dense representation is guarded by an explicit live-element limit and is only a small-Si oracle; GaAs must use a later streamed momentum/localized kernel.

**Tech Stack:** Python 3, NumPy/SciPy, existing EXXCMP1 streaming I/O and BvK key conventions, existing `occupied_factor`, unittest, server-66 LibRI replay.

---

## File map

- Create `tools/exx_thc/src/exx_thc/supercell.py`: finite-BvK layouts, local-RI pair expansion, dense/occupied exchange, H block extraction, and memory guard.
- Create `tools/exx_thc/tests/test_supercell.py`: layout, four-anchor expansion, direct/occupied equivalence, extraction, and error cases.
- Modify `tools/exx_thc/src/exx_thc/cli.py`: add `supercell-gate` without changing `compare` or `project` behavior.
- Modify `tools/exx_thc/tests/test_cli.py`: command success, production-H mismatch, memory guard, scalar, and no-output-on-failure tests.
- Modify `tools/exx_thc/docs/si_real64_gate_2026-08-11.md`: append the exact dense-oracle result and next stop decision.
- Modify `docs/superpowers/specs/2026-08-10-mps-exx-tensor-compression-design.md`: mark the finite-supercell oracle as the Stage-A definition of exactness.

### Task 1: Finite-supercell layout and map expansion

**Files:**
- Create: `tools/exx_thc/src/exx_thc/supercell.py`
- Create: `tools/exx_thc/tests/test_supercell.py`

- [ ] **Step 1: Write the failing layout and expansion tests**

Use a one-atom period `(2,1,1)` with one auxiliary and one AO. Put C values 2 at R=0 and 3 at R=-1, and use compatible one-dimensional V/D maps. The expansion must replicate translations and add each local-RI coefficient in both ordered AO positions:

```python
class SupercellAssemblyTest(unittest.TestCase):
    def test_c_expansion_contains_both_ao_orders_and_all_cells(self):
        period = (2, 1, 1)
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[[2.0]]]),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[[3.0]]]),
        }
        matrix_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[5.0]]),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[7.0]]),
        }
        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, period
        )
        coefficient = assemble_pair_coefficient(
            coefficient_blocks, layout, max_elements=64
        )
        expected = np.zeros((2, 2, 2))
        for cell in range(2):
            expected[cell, cell, cell] += 4.0
            other = (cell - 1) % 2
            expected[cell, cell, other] += 3.0
            expected[cell, other, cell] += 3.0
        np.testing.assert_array_equal(coefficient, expected)

    def test_v_and_d_expansion_follow_outer_cell_plus_r(self):
        period = (2, 1, 1)
        coefficient_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[[2.0]]]),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[[3.0]]]),
        }
        matrix_blocks = {
            BlockKey(0, 0, (0, 0, 0)): np.asarray([[5.0]]),
            BlockKey(0, 0, (-1, 0, 0)): np.asarray([[7.0]]),
        }
        layout = infer_supercell_layout(
            coefficient_blocks, matrix_blocks, matrix_blocks, period
        )
        matrix = assemble_translation_matrix(
            matrix_blocks, layout, space="ao", max_elements=16
        )
        np.testing.assert_array_equal(matrix, np.asarray([[5.0, 7.0], [7.0, 5.0]]))
```

The R=0 C block contributes twice because the two AO orders coincide; this is intentional for the pair coefficient \(C^{p\text{-anchor}}+C^{q\text{-anchor}}\). This is source-backed rather than a fit: `LRI_CV<Tdata>::DPcal_C_dC` constructs the same-atom/same-cell block as `0.5 * L * A`, while an off-site pair stores the two auxiliary-anchor coefficients in the two reversed map records. Therefore the oracle always adds both AO orders and never changes a factor after seeing the physical result.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=tools/exx_thc/src python -W error -m unittest \
  tools.exx_thc.tests.test_supercell.SupercellAssemblyTest -v
```

Expected: import failure for `exx_thc.supercell`.

- [ ] **Step 3: Implement the layout and expansion**

Define one immutable layout contract and use it for all three maps:

```python
@dataclass(frozen=True)
class SupercellLayout:
    period: tuple[int, int, int]
    cells: tuple[tuple[int, int, int], ...]
    atoms: tuple[int, ...]
    ao_dimensions: Mapping[int, int]
    auxiliary_dimensions: Mapping[int, int]
    ao_offsets: Mapping[tuple[tuple[int, int, int], int], int]
    auxiliary_offsets: Mapping[tuple[tuple[int, int, int], int], int]
    nao_supercell: int
    naux_supercell: int
```

Export these exact typed call contracts: `infer_supercell_layout(C, V, D,
period) -> SupercellLayout`, `assemble_pair_coefficient(blocks, layout,
max_elements) -> np.ndarray`, and `assemble_translation_matrix(blocks, layout,
space, max_elements) -> np.ndarray`.

`space` accepts exactly `"ao"` or `"auxiliary"`. Define immutable offsets for
`(cell, atom)` and enforce consistent per-atom dimensions across C/D and C/V.
Use cell tuples in `np.ndindex(period)` and modulo addition component by
component. Canonicalize every stored R modulo the period before addition, so
translation aliases sum rather than overwrite. The core additions are:

```python
coefficient[aux_slice, first_slice, second_slice] += block
coefficient[aux_slice, second_slice, first_slice] += block.transpose(0, 2, 1)
```

and

```python
matrix[first_slice, second_slice] += block
```

Before allocation, compute element counts with Python integers and raise
`ValueError("dense supercell allocation exceeds max_elements")` when the limit
would be exceeded. Require real64 or complex128 arrays, three C tensor
dimensions, two matrix dimensions, positive period, contiguous atom ids, and
finite values. `max_elements` is a positive integer. Reject unknown `space`,
missing AO/auxiliary dimensions, and incompatible C/V/D shapes before any dense
allocation.

- [ ] **Step 4: Run focused GREEN and full Python regression**

Run the focused command from Step 2, then:

```bash
PYTHONPATH=tools/exx_thc/src python -W error -m unittest discover \
  -s tools/exx_thc/tests -v
```

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit**

```bash
git add tools/exx_thc/src/exx_thc/supercell.py \
        tools/exx_thc/tests/test_supercell.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "feat: assemble finite-cell EXX reference"
```

### Task 2: Direct CVCD and exact occupied contraction

**Files:**
- Modify: `tools/exx_thc/src/exx_thc/supercell.py`
- Modify: `tools/exx_thc/tests/test_supercell.py`

- [ ] **Step 1: Write the failing exactness tests**

Construct a complex random C, Hermitian positive V, and rank-two density
`D=O@O.conj().T`. Test the two formulas element by element:

```python
direct = direct_exchange(coefficient, metric, density)
occupied = occupied_exchange(coefficient, metric, O)
np.testing.assert_allclose(occupied.matrix, direct.matrix, rtol=0.0, atol=1.0e-12)
```

Also test zero density, a zero metric mode, non-Hermitian V/D rejection, shape
conflicts, nonfinite output rejection, and that diagnostic temporary shapes do
not contain a reconstructed four-index ERI.

- [ ] **Step 2: Run the tests and verify RED**

Expected: `direct_exchange` and `occupied_exchange` are undefined.

- [ ] **Step 3: Implement the two contractions without an ERI temporary**

Use the exact conventions:

```python
def direct_exchange(C, V, D):
    left = np.einsum("apq,qr->apr", C, D, optimize=True)
    coulomb = np.einsum("ab,bsr->asr", V, C.conj(), optimize=True)
    H = np.einsum("apr,asr->ps", left, coulomb, optimize=True)
    return ExchangeResult(H, (left.shape, coulomb.shape, H.shape),
                          left.size + coulomb.size + H.size)

def occupied_exchange(C, V, O):
    B = np.einsum("apq,qv->apv", C, O, optimize=True)
    VB = np.einsum("ab,bsv->asv", V, B.conj(), optimize=True)
    H = np.einsum("apv,asv->ps", B, VB, optimize=True)
    return ExchangeResult(H, (B.shape, VB.shape, H.shape),
                          B.size + VB.size + H.size)
```

Promote inputs to C-contiguous complex128 for the dense oracle, use scaled
finite norms to avoid overflow diagnostics, and return the actual temporary
shapes with the matrix:

```python
@dataclass(frozen=True)
class ExchangeResult:
    matrix: np.ndarray
    temporary_shapes: tuple[tuple[int, ...], ...]
    temporary_elements: int
```

Export `direct_exchange(C, V, D) -> ExchangeResult` and
`occupied_exchange(C, V, O) -> ExchangeResult`. `temporary_elements` is the
sum of the simultaneously retained contraction temporaries and H, excluding
shared input arrays. Do not call `np.einsum` with a four-index output. Validate
all shapes, Hermiticity residuals, and finiteness before returning.

- [ ] **Step 4: Run focused and full tests**

Expected: direct and occupied H agree within `1e-12`; the whole suite passes.

- [ ] **Step 5: Commit**

Commit with message `feat: add exact occupied EXX oracle` and the required
Codex/AroundPeking attribution.

### Task 3: Convert dense H to LibRI blocks and reproduce post-2D energy

**Files:**
- Modify: `tools/exx_thc/src/exx_thc/supercell.py`
- Modify: `tools/exx_thc/tests/test_supercell.py`

- [ ] **Step 1: Write failing extraction and energy tests**

For every atom pair and canonical cell displacement, extract rows in reference
cell zero and columns in the displaced cell. The extracted map must round-trip
to the corresponding block-circulant matrix. Define energy exactly as LibRI
`Exx_Post_2D::cal_energy`: sum `np.vdot(D_block, H_block)` only over matching
keys, with the raw D map as the first argument.

- [ ] **Step 2: Verify RED**

Expected: missing `extract_reference_cell_blocks` and `map_dotc`.

- [ ] **Step 3: Implement extraction and dotc**

Export `extract_reference_cell_blocks(matrix, layout, scalar) -> TensorMap` and
`map_dotc(density_blocks, h_blocks) -> complex`. Return canonical keys for all
BvK translations. With `scalar="real64"`, require the imaginary residual to be
at most `1e-12*max(real_scale,1)` and return float64 blocks; with
`scalar="complex128"`, retain complex128. Reject any other scalar, shape or key
intersections whose tensor shapes differ.
`map_dotc` uses only the exact key intersection and computes
`sum(np.vdot(D[key], H[key]))`, matching `Exx_Post_2D::cal_energy`.

- [ ] **Step 4: Run focused and full tests**

Expected: exact extraction roundtrip, exact synthetic energy, and full suite
green.

- [ ] **Step 5: Commit**

Commit with message `feat: map occupied EXX oracle to LibRI blocks` and the
required attribution.

### Task 4: Add a strict `supercell-gate` command

**Files:**
- Modify: `tools/exx_thc/src/exx_thc/cli.py`
- Modify: `tools/exx_thc/tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

The command is:

```text
python -m exx_thc.cli supercell-gate \
  --C C.active.exxcmp --V V.active.exxcmp --D-full D.full.exxcmp \
  --D-post D.raw.exxcmp --H-reference H.lri.exxcmp \
  --energy-reference E.lri.scalar --period 3 1 1 \
  --H-dense-out H.dense.exxcmp --H-occ-out H.occ.exxcmp \
  --max-elements 100000000
```

Tests require JSON fields `dense_H_rel_fro`, `occupied_H_rel_fro`,
`dense_occ_H_rel_fro`, `dense_E_abs_Ry_atom`, `occupied_E_abs_Ry_atom`,
`occupied_ranks_by_supercell`, `dense_bytes`, `occupied_bytes`, and `pass`.
The command passes only when dense-vs-production and occupied-vs-production H
are each at most `1e-10`, both energies per atom are at most `1e-10`, and
dense-vs-occupied H is at most `1e-12`. On every failure before output
publication, neither H output may exist.

- [ ] **Step 2: Verify RED**

Expected: argparse rejects `supercell-gate`.

- [ ] **Step 3: Implement the command**

Read inputs with the existing streaming reader, require same scalar type and
rank 0 of 1, assemble C/V/D, hermitize D only after recording its residual,
call `occupied_factor(..., 0.0)`, verify `D=O@O.conj().T` within `1e-12`, compute
both H matrices, extract maps, and compute post-2D energies from D.post.
`dense_bytes` and `occupied_bytes` are
`16 * ExchangeResult.temporary_elements`, because both oracle paths promote
their contraction temporaries to complex128; they deliberately exclude shared
C/V/D and serialized input/output maps.

Before assembling, compute with Python integers the live-element upper bounds
for C, V, D, O, direct/occupied temporaries, and H. Reject the command before
allocation if either path exceeds `--max-elements`; the limit is for the sum of
simultaneously live dense arrays, not merely the largest single array.

Publish the two H outputs as one recoverable operation. Require both output
paths to share a parent and not exist. Write each snapshot to a unique temporary
directory in that parent, retain the temporary hard links, then publish with
exclusive `os.link`. If either publication fails, remove only a final whose
device/inode still matches its retained temporary; then remove the temporary
directory. Print strict JSON with `allow_nan=False` only after successful joint
publication.

- [ ] **Step 4: Run CLI tests and full regression**

Expected: all command tests and the complete suite pass under `-W error`.

- [ ] **Step 5: Commit**

Commit with message `feat: gate full EXX with occupied reference` and required
attribution.

### Task 5: Run the small-Si authoritative gate on server 66

**Files:**
- Modify: `tools/exx_thc/docs/si_real64_gate_2026-08-11.md`
- Modify: `docs/superpowers/specs/2026-08-10-mps-exx-tensor-compression-design.md`
- Create remotely: `runs/si_kp_pbe0/supercell_gate_<commit>/`

- [ ] **Step 1: Sync and rebuild with fixed provenance**

Push the feature branch to `/home/ghj/git/abacus-develop-mps-exx.git`, update the
clean remote worktree with old-Git-compatible fetch/reset against the exact
FETCH_HEAD, rebuild the two replay tests, and run CTest plus the full Python
suite. Record commit, executable SHA256, CMakeCache SHA256, LibRI SHA, and zero
missing `ldd` entries.

- [ ] **Step 2: Submit the gate through Slurm**

Use partition 640, account ghj, one node, one rank, one CPU, 8 GB, and 10
minutes. Run the command from Task 4 against snapshot job 407502. Do not execute
the dense oracle on the login node. Preserve stdout, stderr, JSON, H outputs,
and SHA256 manifest even when the gate fails.

- [ ] **Step 3: Apply the hard decision**

- If dense H does not reproduce LibRI, stop and correct only pair-anchor,
  translation, conjugation, or normalization semantics; do not tune occupied
  thresholds.
- If dense H passes but occupied H fails, stop because the factorization or
  contracted-leg formula is wrong.
- If both pass, mark the fixed-leg C-map route as disproved but the exact
  AO--occupied route as mathematically validated.

- [ ] **Step 4: Update the evidence documents**

Append job ids, exact metrics, timings, peak memory, paths, and hashes to the Si
gate note. Update the design Stage-A status without deleting jobs 407502,
407507, or 407508.

- [ ] **Step 5: Commit the evidence**

Commit with message `docs: validate label-aware occupied EXX reference` and the
required attribution.

### Task 6: Decide whether GaAs k444 is worth the next implementation

**Files:**
- Modify: `tools/exx_thc/docs/si_real64_gate_2026-08-11.md`
- Modify: `docs/superpowers/plans/2026-08-10-exx-occupied-thc-feasibility.md`

- [ ] **Step 1: Record the unavoidable periodic storage factor**

For GaAs, use `NAO=82`, nominal `Nocc=14`, and `Nk=64`. A stored two-momentum
AO--occupied tensor has the relative factor

```text
Nk * Nocc / NAO = 64 * 14 / 82 = 10.9268292683
```

against a one-momentum AO--AO C representation before local sparsity. Therefore
the exact occupied transform alone can reduce a contracted AO dimension by
about `82/14=5.857`, but storing every `(k,q)` B tensor would be about 10.9
times larger, not smaller.

- [ ] **Step 2: Apply the next-stage criterion**

Proceed to GaAs EXX only with a design that either streams B without storing all
`(k,q)` sectors or uses localized occupied/Wannier/ISDF factors that retain
LibRI real-space locality. Require a static peak-memory estimate below 180 GB
and a factor-storage estimate at least 10 times smaller than the original C
snapshot. Otherwise report the exact occupied route as a mathematical identity
but not a viable ABACUS/LibRI acceleration.

- [ ] **Step 3: Update the main plan**

Replace the invalid fixed-leg replay gate with the passed dense oracle and the
streamed/localized precondition. Do not submit GaAs merely because the small-Si
identity passes.

- [ ] **Step 4: Commit**

Commit the decision with message `docs: assess periodic occupied EXX storage`
and the required attribution.

---

## Self-review checklist

- The plan covers all four LibRI labels through the summed local-RI pair
  coefficient rather than a fixed tensor leg.
- The exact occupied identity is checked against full H and post-2D energy, not
  only an internal C-D-C product.
- Dense allocation has an explicit guard and is never proposed for GaAs.
- The server computation remains Slurm-only and preserves failed results.
- real64 and complex128 types remain covered.
- There are no unspecified production optimizations: GaAs requires a separate
  streamed/localized design after the exact oracle passes.
