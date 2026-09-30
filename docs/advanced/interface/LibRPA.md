# LibRPA reader-v1 SOC symmetry export

For an LCAO RPA export consumed by a reader-v1 LibRPA build, set:

```text
basis_type                  lcao
rpa                         1
out_librpa_reader_version   1
```

SOC symmetry export additionally requires `nspin 4`, `lspinorb 1`, and
`symmetry 1`. Magnetic systems must define the intended magnetic moments in
`STRU`.

## `stru_out.txt` symmetry block

After the existing lattice, atom, and k-point records, ABACUS writes the
spatial operation table when symmetry is enabled:

```text
N row
<9 integer rotation entries and 3 fractional translations, repeated N times>
```

The same 12-field spatial rows are used for ordinary space groups and magnetic
groups. A magnetic group's antiunitary spatial operations are written after its
unitary operations. For `nspin=4`, ABACUS appends:

```text
spin_symmetry <grey_group> 2
<antiunitary flag for each spatial operation>
```

`grey_group` is `1` for a nonmagnetic SOC calculation and `0` for a magnetic
group. The `2` indicates that LibRPA reconstructs the spin rotation from the
spatial operation; ABACUS does not export the old `symrot_*` sidecar files as
part of this reader-v1 interface.
