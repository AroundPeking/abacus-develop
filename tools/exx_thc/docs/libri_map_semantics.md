# LibRI EXX tensor-map and replay semantics

## Audit provenance

This note records the source contract used by the serial replay and occupied-space projection tools. The ABACUS audit was made in this repository on branch `codex/mps-exx-k444`, starting from commit `ab177743bedd833ee74a28ffecf94763d302c61e` (2026-08-10). The EXXCMP1 map type and reader/writer are in [`exx_compression_io.h` at that fixed ABACUS revision](https://github.com/AroundPeking/abacus-develop/blob/ab177743bedd833ee74a28ffecf94763d302c61e/source/source_lcao/module_ri/exx_compression_io.h#L17-L24), especially `ExxCompressionIO::TensorMap`, `write_map`, and `read_map`.

The matching local LibRI source audit used the external dependency checkout `/Users/ghj/code/LibRI`, branch `codex/cd-weighted-short-screening`, commit `21f92f943bf2f284fee9128bcd3e1a9f197916e1`. The relevant external files are:

- [`include/RI/physics/Exx.hpp`, lines 54-108](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/physics/Exx.hpp#L54-L108): `set_Cs`, `set_Vs`, and `set_Ds` label assignments;
- [`include/RI/ri/LRI-set.hpp`, lines 18-73](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-set.hpp#L18-L73): `LRI::set_parallel`, period storage, flags, and `LRI::set_tensors_map2`;
- [`include/RI/ri/LRI-cal_loop3.hpp`, lines 108-117](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-cal_loop3.hpp#L108-L117): `LRI::cal_loop3` entry;
- [`include/RI/physics/Exx_Post_2D.hpp`, lines 45-81](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/physics/Exx_Post_2D.hpp#L45-L81): `Exx_Post_2D::set_tensors_map2` and `cal_energy`.

This is verified behavior for that exact local dependency revision, not a claim about every past or future LibRI release. Task 8 must recheck the dependency revision used by the authoritative server build.

## Map keys, labels, and block shapes

All four objects use the nested map

```text
map<ia1, map<(ia2, R), Tensor<Scalar>>>
```

where `R` has three integer components.

`Scalar` is `double` for the nspin=1 real path and `complex<double>` for the
complex path. A replay requires C, V, D, and, when present, D.post to use the
same EXXCMP1 scalar tag; it dispatches to the matching `RI::Exx` template.

| Object | LibRI labels | Block shape |
| --- | --- | --- |
| `C` | `a`, `b` | `[naux(ia1), nao(ia1), nao(ia2)]` |
| `V` | `a0b0` | `[naux(ia1), naux(ia2)]` |
| `D` | `a1b1`, `a1b2`, `a2b1`, `a2b2` | `[nao(ia1), nao(ia2)]` |
| `H` | result map `RI::Exx::Hs` produced by `RI::Exx::cal_Hs` | AO-pair matrix blocks |

The label assignments are explicit in the fixed-revision LibRI [`Exx::set_Cs`, `set_Vs`, and `set_Ds` definitions](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/physics/Exx.hpp#L54-L108). ABACUS writes the loaded active pools in `Exx_LRI.hpp`: [`V.active`, lines 1601-1615](https://github.com/AroundPeking/abacus-develop/blob/ab177743bedd833ee74a28ffecf94763d302c61e/source/source_lcao/module_ri/Exx_LRI.hpp#L1601-L1615), [`C.active`, lines 1694-1722](https://github.com/AroundPeking/abacus-develop/blob/ab177743bedd833ee74a28ffecf94763d302c61e/source/source_lcao/module_ri/Exx_LRI.hpp#L1694-L1722), and [`D.raw`/`D.active` plus `cal_Hs`, lines 2287-2308](https://github.com/AroundPeking/abacus-develop/blob/ab177743bedd833ee74a28ffecf94763d302c61e/source/source_lcao/module_ri/Exx_LRI.hpp#L2287-L2308).

`H` in the replay is exactly the un-postprocessed `RI::Exx::cal_Hs` map. Its scalar energy is the value computed by `Exx_Post_2D::cal_energy(D,H)`: see fixed-revision [`Exx::cal_Hs`, lines 297-326](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/physics/Exx.hpp#L297-L326) and [`Exx_Post_2D::cal_energy`, lines 53-81](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/physics/Exx_Post_2D.hpp#L53-L81). The text output always records `real imag`; the imaginary field is zero on the real path. A critical detail is visible in [`Exx::set_Ds`, lines 94-108](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/physics/Exx.hpp#L94-L108): the density passed into `lri.set_tensors_map2` becomes the active contraction map used for `H`, while `post_2D.saves["Ds_"]` is made from the original incoming density and is later used for energy. Production energy therefore does **not** use the loaded active density map.

## Period and loaded-map state

`period` is `RI::Exx::lri.period`. [`LRI::set_parallel` stores it and `LRI::set_tensors_map2` applies the period/communication/filter flags](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-set.hpp#L18-L73). When `flag_period=true`, translation aliases are summed modulo that period. Once a map has passed through this operation, `R` is interpreted modulo the configured period by the later [`cal_loop3` contractions](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-cal_loop3.hpp#L108-L117).

The replay distinguishes two states:

- **Active/final map:** the snapshot already is the map stored in a LibRI data pool after production periodization, communication, and filtering. Replay must not repeat any of those operations. Active `C`, `V`, and `D_path` use `flag_period=false`, `flag_comm=false`, and `flag_filter=false`. `D_path` is used only for the `H` contraction. The separately required `D_post_path` is the matching raw density snapshot and is passed only to `post_2D.set_tensors_map2` for the energy contraction.
- **Raw density map:** `D_path` is the ABACUS input density before LibRI map preparation. Replay preserves that original map as the `post_2D` energy source, and separately applies periodization exactly once for the `H` contraction, with `flag_period=true`, `flag_comm=false`, and `flag_filter=false`. The resulting `data_pool["Ds_"].Ds_ab` is written as `D.full`; that exact map, rather than `D.raw`, is the only density accepted by the Python PSD and occupied-space analysis. `D_post_path` is forbidden in raw mode because `D_path` already supplies both branches.

Production threshold filtering is deliberately absent from replay. A replay compares a frozen state; changing its sparsity would change the object being tested. The same reason requires active `C` and `V` to use the explicit three-false flags even though LibRI's `set_tensors_map2` defaults are periodization and filtering enabled.

The Python Fourier gate first converts each sparse R map to k sectors with `to_k`. That operation sums all R aliases modulo `period` and treats absent R blocks as zero. It then compares those normalized k sectors with `to_k(from_k(...))`, one atom pair at a time to avoid retaining a second full snapshot; it does not compare the original sparse key set with the full-period zero blocks created by `from_k`. `C` and `D` roundtrip errors are reported separately, their maximum is also reported, and both must be below `1e-13`. All transformed, roundtrip, projected-matrix, and final projected-snapshot values are checked for finiteness, because summing individually finite translation aliases can still overflow. This preserves the Task 4 transform convention and its tested phase choice while retaining physical sparse-map semantics. If any Fourier, Hermiticity, or PSD gate fails, projection allocates no output map and reports zero output blocks and bytes.

## Current occupied-factor boundary

At LibRI commit `21f92f943bf2f284fee9128bcd3e1a9f197916e1`, the public `RI::Exx` path accepts a density map through [`set_Ds`](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/physics/Exx.hpp#L94-L108); it does not accept an occupied factor `O`, its pseudoinverse, or an occupied-space projector. `Exx::cal_Hs` passes the four density labels into [`LRI::cal_loop3`](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-cal_loop3.hpp#L108-L117), whose public signature receives label choices and an output map. No occupied-factor input is exposed there.

Consequently, `exx_thc.cli project` is an offline diagnostic: it builds `C_occ(k) = C(k) O(k) O(k)^+`, writes a new canonical full-period `C` snapshot, and verifies the density-contracted product. It does not claim that current LibRI can consume `O` directly. This conclusion is revision-specific and must be re-audited if the dependency changes.

It also is not a zero-error replacement for the complete LibRI `cal_Hs`. The
four EXX labels connect D to `a1` or `a2` on each C tensor. A fixed projection
of the stored third tensor dimension therefore projects a contracted leg in
some labels and a free H leg in others. The server-66 small-Si gate recorded in
[`si_real64_gate_2026-08-11.md`](si_real64_gate_2026-08-11.md) preserves the
energy to rounding but changes H by a relative Frobenius norm of about 0.156.
The diagnostic remains useful for Fourier, PSD, occupied-rank, and
single-orientation algebra checks; full-H validation requires a label-aware or
two-momentum AO--occupied kernel.

The four fixed-revision contraction bodies are
[`a0b0_a1b1`](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-cal_loop3.hpp#L181-L279),
[`a0b0_a1b2`](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-cal_loop3.hpp#L347-L445),
[`a0b0_a2b1`](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-cal_loop3.hpp#L641-L742),
and
[`a0b0_a2b2`](https://github.com/AroundPeking/LibRI/blob/21f92f943bf2f284fee9128bcd3e1a9f197916e1/include/RI/ri/LRI-cal_loop3.hpp#L941-L1047).
For a dense finite-supercell oracle, each anchored C block must populate both
AO orders. The apparent on-site double addition is source-defined:
[`LRI_CV<Tdata>::DPcal_C_dC`](https://github.com/AroundPeking/abacus-develop/blob/ab177743bedd833ee74a28ffecf94763d302c61e/source/source_lcao/module_ri/LRI_CV.hpp#L329-L421)
stores the same-atom/same-cell coefficient with an explicit factor `0.5` and
stores the two off-site auxiliary-anchor coefficients in reversed map records.
The oracle must therefore use that normalization unchanged, not fit a factor to
the production H result.

The global projector can create a block that was absent from the input `C` map. The projected output therefore contains every auxiliary-row atom present in `C` paired with every AO-column atom present in `D`. Missing input pairs enter global `C(k)` as zero blocks, but their projected output blocks are retained (including blocks that remain zero) for every canonical BvK translation. Dropping those pairs after the global matrix multiplication would not in general preserve `C D C^dagger`.

The PSD gate is relative to the largest positive density eigenvalue in each k sector: `lambda_min >= -1e-10 * lambda_max`. If `lambda_max` is zero, the zero matrix passes and any negative eigenvalue fails; the scaled-minimum JSON diagnostic is `null` when no positive scale exists for a negative eigenvalue. This gate is applied before the separate numerical floor inside `occupied_factor`.

## ABACUS postprocessing boundary

Replay `H` and `E` are the LibRI-side values before ABACUS spin/unit postprocessing. ABACUS later multiplies every `H` block by `-2` in `Exx_LRI::post_process_Hexx` and multiplies the accumulated energy by `-SPIN_multiple` in `post_process_Eexx`; see the fixed ABACUS revision [`Exx_LRI.hpp`, lines 2398-2415](https://github.com/AroundPeking/abacus-develop/blob/ab177743bedd833ee74a28ffecf94763d302c61e/source/source_lcao/module_ri/Exx_LRI.hpp#L2398-L2415). These transformations must not be folded into replay, because doing so would mix the LibRI contraction gate with the ABACUS Hamiltonian and spin boundary.

## Replay configuration

The command-line replay executable accepts one configuration path. The format is one strict key followed by values per line; blank lines and lines beginning with `#` are ignored. Paths consume the remainder of the line, so spaces are supported. Fixed keys cannot be repeated, atom identifiers cannot be repeated, and unknown or missing keys are errors.

```text
period 2 1 1
lattice_1 1 0 0
lattice_2 0 1 0
lattice_3 0 0 1
atom 0 0 0 0
C_path /path/to/C.active.exxcmp
V_path /path/to/V.active.exxcmp
D_path /path/to/D.active.exxcmp
D_state active
D_post_path /path/to/D.raw.exxcmp
H_out /path/to/H.lri.exxcmp
E_out /path/to/E.lri.scalar
```

For raw replay, replace the three density lines above with:

```text
D_path /path/to/D.raw.exxcmp
D_state raw
D_full_out /path/to/D.full.exxcmp
```

`atom` may occur once for each distinct integer atom id. `D_state` is exactly `active` or `raw`; `D_post_path` is required only for active, and `D_full_out` is required only for raw. All configured input and output paths must be distinct. Every input must be a complete serial EXXCMP1 real64 or complex128 snapshot (`rank=0`, `nranks=1`) with finite tensor values, and all inputs in one replay must have the same scalar type. The executable also requires MPI world size one, so distributed shard files cannot accidentally be treated as a complete map. Python comparison accepts either scalar type but requires reference and candidate to match. Occupied projection likewise requires matching C/D.full types and preserves that type; for real64 it rejects a non-negligible inverse-transform imaginary component before writing the real output.

Final output paths are never overwritten. Replay creates every temporary with same-directory `mkstemp`, so concurrent processes or shared-filesystem nodes cannot truncate one another's temporary files. It publishes each final through an exclusive POSIX link and publishes `E_out` last as the completion marker. The temporary hardlinks remain present until the whole final set has been linked; if a caught failure occurs during publication, replay compares the live temp/final inode identities, removes only finals published by that process, and then cleans its temporary files. A process crash can leave an incomplete set without `E_out`; a later replay rejects every existing final rather than mixing new files with that set. Consumers must therefore require both `H_out` and `E_out`, plus `D_full_out` for raw replay, before treating a replay as complete.
