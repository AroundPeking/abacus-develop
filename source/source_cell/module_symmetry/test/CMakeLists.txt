remove_definitions(-D__LCAO)
remove_definitions(-D__MLALGO)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
AddTest(
  TARGET MODULE_CELL_SYMMETRY_analysis
  LIBS parameter base ${math_libs} device symmetry 
  SOURCES symmetry_test.cpp symmetry_test_analysis.cpp
)
AddTest(
  TARGET MODULE_CELL_SYMMETRY_symtrz
  LIBS parameter base ${math_libs} device symmetry 
  SOURCES symmetry_test.cpp symmetry_test_symtrz.cpp
)
AddTest(
  TARGET MODULE_CELL_SYMMETRY_rotation_spin
  LIBS parameter base ${math_libs} device
  SOURCES symmetry_rotation_spin_test.cpp ../symmetry_rotation_spin.cpp
)
AddTest(
  TARGET MODULE_CELL_SYMMETRY_rho_soc
  LIBS parameter base ${math_libs} device symmetry
  SOURCES symmetry_rho_soc_test.cpp
    ${ABACUS_SOURCE_DIR}/source_estate/module_dm/density_matrix.cpp
    ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/base_matrix.cpp
    ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/hcontainer.cpp
    ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/atom_pair.cpp
    ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
)
