remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
remove_definitions(-D__MPI)

add_library(
    io_input_serial
    OBJECT
    ../read_input_item_system.cpp
    ../read_input_item_elec_stru.cpp
    ../read_input_item_relax.cpp
    ../read_input_item_md.cpp
    ../read_input_item_ofdft.cpp
    ../read_input_item_sdft.cpp
    ../read_input_item_tddft.cpp
    ../read_input_item_deepks.cpp
    ../read_input_item_model.cpp
    ../read_input_item_postprocess.cpp  
    ../read_input_item_exx_dftu.cpp
    ../read_input_item_other.cpp   
    ../read_input_item_output.cpp
    ../read_input.cpp
    ../bcast_globalv.cpp
)

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
AddTest(
  TARGET io_read_input_serial
  LIBS ${math_libs} io_input_serial parameter
  SOURCES read_input_test.cpp ../../module_base/test/tool_quit_no_exit.cpp
)

AddTest(
  TARGET io_read_item_serial
  LIBS ${math_libs} base device io_input_serial parameter
  SOURCES read_input_item_test.cpp
)

AddTest(
  TARGET io_read_input_tool
  SOURCES read_input_tool_test.cpp
)

AddTest(
  TARGET io_rho_io
  LIBS ${math_libs} base device cell_info parameter
  SOURCES rho_io_test.cpp ../read_cube.cpp ../write_cube.cpp ../read_rho.cpp ../output.cpp
)

AddTest(
  TARGET io_dmk_io
  LIBS ${math_libs} base device cell_info parameter
  SOURCES io_dmk_test.cpp ../io_dmk.cpp ../output.cpp ../../module_basis/module_ao/parallel_2d.cpp
)

AddTest(
  TARGET io_nscf_band
  LIBS ${math_libs} base device
  SOURCES nscf_band_test.cpp ../nscf_band.cpp
)

AddTest(
  TARGET io_read_wfc_nao_test
  LIBS ${math_libs} base device
  SOURCES read_wfc_nao_test.cpp ../read_wfc_nao.cpp ../../module_psi/psi.cpp ../../module_basis/module_ao/parallel_orbitals.cpp ../../module_basis/module_ao/parallel_2d.cpp
)
