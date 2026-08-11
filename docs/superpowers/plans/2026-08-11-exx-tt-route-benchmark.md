# EXX TT Route Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure all six TT index orders for static `C`, occupied `B`, and Coulomb-whitened `X`, then time the best accuracy-qualified route using direct TT-core EXX contraction.

**Architecture:** Add small TT-network primitives beside the existing stable three-index TT-SVD, then build a benchmark layer that owns permutation scans, numerical errors, route accounting, and winner selection. Extend the existing snapshot CLI only as an adapter from LibRI snapshots to dense label-aware supercell arrays; all tensor algebra remains independently testable without file I/O.

**Tech Stack:** Python 3.8, NumPy 1.23.3, SciPy 1.9.1, `unittest`, existing `exx_thc` snapshot/supercell/metric modules, Slurm on server 66.

---

### Task 1: Direct TT-core transforms and Gram contraction

**Files:**
- Modify: `tools/exx_thc/src/exx_thc/tt.py`
- Modify: `tools/exx_thc/tests/test_tt.py`

- [ ] **Step 1: Write failing tests for core transforms and all output-axis positions**

Add imports for `tt_core_elements`, `tt_mode_transform`, and `tt_gram`, then add tests based on a deterministic complex tensor:

```python
def test_mode_transform_matches_dense_for_every_axis(self):
    rng = np.random.default_rng(811)
    tensor = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
    for axis in range(3):
        transform = rng.normal(size=(5, tensor.shape[axis])) + 1j * rng.normal(
            size=(5, tensor.shape[axis])
        )
        tt = tt_svd_3(tensor, 0.0)
        transformed = tt_mode_transform(tt, axis, transform)
        expected = np.tensordot(transform, tensor, axes=(1, axis))
        expected = np.moveaxis(expected, 0, axis)
        np.testing.assert_allclose(transformed.reconstruct(), expected, rtol=2e-13, atol=2e-13)

def test_gram_matches_dense_for_every_output_axis(self):
    rng = np.random.default_rng(812)
    tensor = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
    for axis in range(3):
        tt = tt_svd_3(tensor, 0.0)
        flattened = np.moveaxis(tensor, axis, 0).reshape(tensor.shape[axis], -1)
        np.testing.assert_allclose(
            tt_gram(tt, axis), flattened @ flattened.conj().T, rtol=2e-13, atol=2e-13
        )
```

Also test invalid axes, transform shapes, non-finite transforms, zero-rank TT, and exact core-element accounting.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=tools/exx_thc/src /Users/ghj/apps/anaconda3/bin/python -W error \
  -m unittest tools.exx_thc.tests.test_tt
```

Expected: import failure because the three new public functions do not exist.

- [ ] **Step 3: Implement minimal TT primitives**

Use a helper that replaces only the selected physical core:

```python
def tt_mode_transform(tt: TT3, axis: int, transform: np.ndarray) -> TT3:
    axis = _physical_axis(axis)
    matrix = np.asarray(transform, dtype=np.complex128)
    physical_shapes = (tt.g1.shape[0], tt.g2.shape[1], tt.g3.shape[1])
    if matrix.ndim != 2 or matrix.shape[1] != physical_shapes[axis]:
        raise ValueError("TT mode transform has incompatible shape")
    if not np.isfinite(matrix).all():
        raise ValueError("TT mode transform must contain only finite values")
    cores = [tt.g1, tt.g2, tt.g3]
    if axis == 0:
        cores[0] = np.einsum("pi,ia->pa", matrix, cores[0])
    elif axis == 1:
        cores[1] = np.einsum("pj,ajb->apb", matrix, cores[1])
    else:
        cores[2] = np.einsum("pk,bk->bp", matrix, cores[2])
    return TT3(cores[0], cores[1], cores[2], tt.spectra, tt.discarded_weights,
               tt.error_bound, tt.ranks)
```

Implement `tt_gram` through left and right environments so the dense
rank-three tensor is never formed.

- [ ] **Step 4: Run focused and full tests for GREEN**

Run the focused command above, then:

```bash
PYTHONPATH=tools/exx_thc/src /Users/ghj/apps/anaconda3/bin/python -W error \
  -m unittest discover -s tools/exx_thc/tests -p 'test_*.py'
```

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit Task 1**

```bash
git add tools/exx_thc/src/exx_thc/tt.py tools/exx_thc/tests/test_tt.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat: contract EXX directly from TT cores'
```

### Task 2: Route scan, accounting, and deterministic winner selection

**Files:**
- Create: `tools/exx_thc/src/exx_thc/tt_benchmark.py`
- Create: `tools/exx_thc/tests/test_tt_benchmark.py`
- Modify: `tools/exx_thc/src/exx_thc/metrics.py`
- Modify: `tools/exx_thc/tests/test_metrics.py`

- [ ] **Step 1: Write failing route-scan tests**

Construct a deterministic low-rank complex `C`, PSD `V`, and occupied factor `O`. Require:

```python
result = scan_exx_tt_routes(C, V, O, tolerances=(0.0, 1.0e-8), repeats=2)
self.assertEqual({point.route for point in result.points}, {"C", "B", "X"})
self.assertEqual(
    {point.order for point in result.points},
    set(itertools.permutations((0, 1, 2))),
)
for point in result.points:
    self.assertTrue(np.isfinite(point.h_rel_fro))
    self.assertGreaterEqual(point.compression_ratio, 0.0)
```

Add focused tests that:

- compare each `relative_tol=0` core contraction against `occupied_exchange(C,V,O).matrix`;
- prove the static `C` route excludes TT-SVD from `exx_seconds` while `B` and `X` include their per-update setup in `steady_seconds`;
- verify winner selection first applies `H<=1e-8`, then time, then the 10% storage tie-break, then static `C`;
- reject invalid tolerances/repeat counts/shapes/non-finite values;
- serialize all report values with `json.dumps(..., allow_nan=False)`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=tools/exx_thc/src /Users/ghj/apps/anaconda3/bin/python -W error \
  -m unittest tools.exx_thc.tests.test_tt_benchmark
```

Expected: `ModuleNotFoundError: No module named 'exx_thc.tt_benchmark'`.

- [ ] **Step 3: Implement route construction and one scan point**

First extend `WhitenedTensor` with a `transform` array equal to the existing
standard whitening transform `factor.conj().T`, shape
`(n_active_aux, n_aux)`, and test
`result.tensor == einsum("xm,miv->xiv", result.transform, cbar)`. This reuses
the exact eigensystem, clipping, and zero-mode decision already made by
`whiten`; the benchmark must not diagonalize `V` a second time. For the LibRI
EXX convention, use `result.transform.conj() == factor.T`, because the target
is `B.T @ V @ B.conj()`, not the standard `B.conj().T @ V @ B`.

Canonical tensors are:

```python
B = np.einsum("mij,jv->miv", C, O, optimize=True)
whitened = whiten(V, B)
sqrt_v = whitened.transform.conj()
X = np.einsum("xm,miv->xiv", sqrt_v, B, optimize=True)
```

For each order, transpose the canonical tensor, run `tt_svd_3`, and map the canonical free-AO axis into the permuted axis. Route-specific direct contractions are:

```python
# C route: reusable C-TT; per update transform j with O.T and mu with sqrt(V).
working = tt_mode_transform(tt_c, order.index(2), O.T)
working = tt_mode_transform(working, order.index(0), sqrt_v)
H = tt_gram(working, order.index(1))

# B route: transform mu with sqrt(V), then contract.
working = tt_mode_transform(tt_b, order.index(0), sqrt_v)
H = tt_gram(working, order.index(1))

# X route: X is already metric transformed.
H = tt_gram(tt_x, order.index(1))
```

Use `time.perf_counter_ns()` and report seconds as finite floats. `compression_ratio` is `dense_elements / core_elements`; spectra are converted to finite Python lists.

- [ ] **Step 4: Implement complete scan and selection**

Create immutable result records with `to_json_dict()`. Precompute `B`, `sqrt_v`, and `X` once per scan, but charge route timing according to the design document. Warm each operation once and report the median of `repeats` samples. Selection filters the accuracy window before comparing time.

- [ ] **Step 5: Run focused and full tests for GREEN**

Run the Task 2 focused command and the full discovery command from Task 1.

- [ ] **Step 6: Commit Task 2**

```bash
git add tools/exx_thc/src/exx_thc/tt_benchmark.py \
        tools/exx_thc/tests/test_tt_benchmark.py \
        tools/exx_thc/src/exx_thc/metrics.py \
        tools/exx_thc/tests/test_metrics.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat: compare EXX TT compression routes'
```

### Task 3: Snapshot-backed `supercell-tt-scan` command

**Files:**
- Modify: `tools/exx_thc/src/exx_thc/cli.py`
- Modify: `tools/exx_thc/tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Reuse the existing `SupercellGateCommandTest` snapshot fixture. Add `supercell-tt-scan` tests requiring:

```python
status, report = self.run_cli(
    "supercell-tt-scan",
    "--C", self.C,
    "--V", self.V,
    "--D-full", self.D_full,
    "--D-post", self.D_post,
    "--H-reference", self.H_reference,
    "--energy-reference", self.energy_reference,
    "--period", "2", "1", "1",
    "--relative-tol", "0", "1e-8",
    "--repeats", "2",
    "--max-elements", "1000000",
)
self.assertIn(status, (0, 1))
self.assertEqual(len(report["points"]), 3 * 6 * 2)
self.assertIn("selected", report)
```

Also require read-before-allocation memory gates, same-scalar checks, exact layout validation, strict finite JSON, and no output files.

- [ ] **Step 2: Run the focused CLI test and verify RED**

Run:

```bash
PYTHONPATH=tools/exx_thc/src /Users/ghj/apps/anaconda3/bin/python -W error \
  -m unittest tools.exx_thc.tests.test_cli.SupercellTTScanCommandTest
```

Expected: argparse rejects `supercell-tt-scan` as an invalid choice.

- [ ] **Step 3: Add the parser and snapshot adapter**

The command accepts the same five snapshots, period, energy reference, and allocation bound as `supercell-gate`, plus repeatable tolerances and repeat count. The adapter reuses existing snapshot/layout validation, assembles `C`, `V`, and `D.full`, obtains `O=occupied_factor(D,0).O`, and calls `scan_exx_tt_routes`.

After each point produces a dense `H` only for validation, map it back with `extract_reference_cell_blocks`, evaluate the existing `_relative_frobenius_union` and `_real_map_energy`, and attach LibRI `H` and Ry/atom errors. Do not publish candidate snapshots.

- [ ] **Step 4: Run CLI, numerical, and full tests for GREEN**

Run the focused CLI test, Task 2 focused test, and full discovery command.

- [ ] **Step 5: Commit Task 3**

```bash
git add tools/exx_thc/src/exx_thc/cli.py tools/exx_thc/tests/test_cli.py
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'feat: scan TT routes from EXX snapshots'
```

### Task 4: Local verification and benchmark artifact

**Files:**
- Modify: `tools/exx_thc/docs/si_real64_gate_2026-08-11.md`
- Create: `tools/exx_thc/docs/si_tt_scan_2026-08-11.json`

- [ ] **Step 1: Run fresh full verification**

```bash
PYTHONPATH=tools/exx_thc/src /Users/ghj/apps/anaconda3/bin/python -W error \
  -m unittest discover -s tools/exx_thc/tests -p 'test_*.py'
git diff --check
```

Expected: zero failures, zero warnings, and clean diff check.

- [ ] **Step 2: Run the small-Si scan from the validated snapshot directory**

Use the exact snapshot root and filenames from the validated server run. Pin
BLAS threads to one for comparable algorithm timings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=tools/exx_thc/src /Users/ghj/apps/anaconda3/bin/python -W error \
  -m exx_thc.cli supercell-tt-scan \
  --C C.active.exxcmp --V V.active.exxcmp \
  --D-full D.full.exxcmp --D-post D.raw.exxcmp \
  --H-reference H.lri.exxcmp --energy-reference E.lri.scalar \
  --period 3 1 1 --relative-tol 0 1e-10 1e-8 1e-6 1e-4 1e-3 \
  --repeats 5 --max-elements 100000000 \
  > si_tt_scan_2026-08-11.json
```

Copy the exact resulting JSON into the tracked artifact path; do not hand-edit numerical fields.

- [ ] **Step 3: Summarize the selected route without overstating Python timing**

Append a table containing route/order/tolerance/ranks/compression/`H` error/energy error/setup/steady EXX. State explicitly whether the 5x compression and 2x speed gates are met.

- [ ] **Step 4: Commit the local result**

```bash
git add tools/exx_thc/docs/si_real64_gate_2026-08-11.md \
        tools/exx_thc/docs/si_tt_scan_2026-08-11.json
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'docs: record small-Si EXX TT scan'
```

### Task 5: Server-66 timing and GaAs decision

**Files:**
- Modify: `tools/exx_thc/docs/si_real64_gate_2026-08-11.md`

- [ ] **Step 1: Sync the exact committed branch to server 66**

Use the existing remote bare-repository route and verify that the remote source HEAD equals the local commit before submission. Record the source SHA in the job log.

- [ ] **Step 2: Submit a one-node, one-process authoritative timing job**

Run the same command and thread pins as Task 4. Capture Slurm elapsed time and MaxRSS in addition to the JSON report. Do not submit GaAs concurrently.

- [ ] **Step 3: Inspect outputs and apply the decision rule**

Require a finished scheduler state, valid strict JSON, finite values, and exact source SHA. Select the fastest accuracy-qualified point using the implemented deterministic rule.

- [ ] **Step 4: Stop or continue**

If either compression is below 5x or route-correct steady speedup is below 2x, document that global TT/MPS is stopped and do not submit GaAs. If all gates pass, prepare a separate bounded GaAs plan that respects the 180 GB memory gate; this task does not itself submit GaAs.

- [ ] **Step 5: Commit the server result and verify attribution**

```bash
git add tools/exx_thc/docs/si_real64_gate_2026-08-11.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'docs: decide EXX TT route from server timing'
git log -1 --format='%H%nAuthor: %an <%ae>%nCommitter: %cn <%ce>%n%s'
git show --check --stat HEAD
```

Expected attribution: author `Codex <codex@openai.com>`, committer `AroundPeking <gonghuanjing@iphy.ac.cn>`.
