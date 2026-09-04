# ABACUS LibRPA Thermal Producer Implementation Plan

**Goal:** Export an unambiguous Fermi-Dirac temperature contract for LibRPA and
correct the Kelvin convenience input so that it produces the requested
physical electronic temperature.

**Architecture:** Keep `band_out` backward compatible and write one versioned
sidecar only for Fermi-Dirac RPA producer runs. Isolate unit conversion and
serialization in small testable helpers used by `RPA_LRI::out_bands`.

---

## Task 1: Correct Kelvin-to-Ry Conversion

**Files:**
- Modify: `source/source_io/module_parameter/read_input_item_elec_stru.cpp`
- Modify: `source/source_io/test_serial/read_input_item_test.cpp`
- Modify: `docs/advanced/input_files/input-main.md`
- Modify: `docs/parameters.yaml`

1. Add a failing input-item test showing that 300 K must produce
   `smearing_sigma = k_B T` in Ry.
2. Replace the current half-`k_B T` conversion by the physical Ry/K constant.
3. Update the generated documentation source and prose.
4. Verify direct `smearing_sigma` parsing remains unchanged.

## Task 2: Add the Thermal Metadata Helper

**Files:**
- Add: `source/source_lcao/module_ri/librpa_thermal_metadata.h`
- Add: `source/source_lcao/module_ri/librpa_thermal_metadata.cpp`
- Add: `source/source_lcao/module_ri/test/librpa_thermal_metadata_test.cpp`
- Modify: `source/source_lcao/module_ri/CMakeLists.txt`
- Modify: `source/source_lcao/module_ri/test/CMakeLists.txt`

1. Add failing tests for Ry-to-Ha conversion, canonical FD method aliases,
   deterministic serialization, invalid dimensions, and non-FD omission.
2. Implement a plain metadata value type and serializer independent of global
   ABACUS state.
3. Write values with 17-digit precision and fixed field names.

## Task 3: Write the Sidecar with Band Data

**Files:**
- Modify: `source/source_lcao/module_ri/RPA_LRI.h`
- Modify: `source/source_lcao/module_ri/RPA_LRI.hpp`
- Modify: `source/source_lcao/module_ri/test/librpa_thermal_metadata_test.cpp`

1. Add a failing fixture test for the complete sidecar next to `band_out`.
2. Call the helper on rank zero after successful band output when the smearing
   method is `fd` or `fermi-dirac`.
3. Use `eferm/2` and `smearing_sigma/2` for Ha metadata, matching `band_out`.
4. Confirm Gaussian, MP, cold, fixed, and default producer runs create no
   thermal sidecar.

## Task 4: Verify Producer Compatibility

1. Run focused input and metadata tests.
2. Build the RPA_LRI object target to catch template integration errors.
3. Run an existing insulating reader-v1 producer smoke test and compare its
   pre-existing files byte-for-byte; the only permitted new file is absent for
   non-FD input.
4. Run a minimal FD producer and validate all sidecar occupations independently
   from `band_out`.

