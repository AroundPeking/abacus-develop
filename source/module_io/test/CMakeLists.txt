remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
remove_definitions(-D__EXX)
remove_definitions(-DUSE_PAW)

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(FILES INPUTs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET io_input_test_para
  LIBS ${math_libs} base device io_input parameter
  SOURCES read_input_ptest.cpp
)

add_test(NAME io_input_test_para_4
      COMMAND mpirun -np 4 ./io_input_test_para
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET read_exit_file_test
  LIBS ${math_libs} base device parameter
  SOURCES read_exit_file_test.cpp ../read_exit_file.cpp
)

add_test(NAME read_exit_file_test_para_4
      COMMAND mpirun -np 4 ./read_exit_file_test
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET io_winput_test
  LIBS ${math_libs} base device
  SOURCES winput_test.cpp ../winput.cpp
)

AddTest(
  TARGET io_output_test
  LIBS ${math_libs} base device
  SOURCES output_test.cpp ../output.cpp
)

AddTest(
  TARGET binstream_test
  SOURCES binstream_test.cpp ../binstream.cpp
)

AddTest(
  TARGET io_write_istate_info_test
  LIBS ${math_libs} base device symmetry parameter
  SOURCES write_istate_info_test.cpp ../write_istate_info.cpp ../output.cpp ../../module_cell/parallel_kpoints.cpp ../../module_cell/klist.cpp
  ../cif_io.cpp
)

AddTest(
  TARGET io_cal_dos
  LIBS ${math_libs} base device
  SOURCES cal_dos_test.cpp ../cal_dos.cpp
)

AddTest(
  TARGET io_write_dos_pw
  LIBS ${math_libs} base device symmetry parameter
  SOURCES write_dos_pw_test.cpp ../cal_dos.cpp ../write_dos_pw.cpp ../output.cpp ../../module_cell/parallel_kpoints.cpp ../../module_cell/klist.cpp
)

AddTest(
  TARGET io_print_info
  LIBS ${math_libs} base device symmetry cell_info parameter
  SOURCES print_info_test.cpp ../print_info.cpp ../output.cpp ../../module_cell/klist.cpp ../../module_cell/parallel_kpoints.cpp
)

AddTest(
  TARGET io_single_R_test
  LIBS ${math_libs}
  SOURCES single_R_io_test.cpp ../single_R_io.cpp
  	../../module_base/global_variable.cpp
	../../module_base/parallel_reduce.cpp
	../../module_base/parallel_common.cpp
	../../module_base/parallel_global.cpp
  ../../module_base/parallel_comm.cpp
)

AddTest(
  TARGET io_write_wfc_nao
  LIBS ${math_libs} base psi device
  SOURCES write_wfc_nao_test.cpp ../write_wfc_nao.cpp ../../module_basis/module_ao/parallel_2d.cpp ../../module_basis/module_ao/parallel_orbitals.cpp ../binstream.cpp
)

install(FILES write_wfc_nao_para.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
find_program(BASH bash)
add_test(NAME io_write_wfc_nao_para
      COMMAND ${BASH} write_wfc_nao_para.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET io_write_orb_info
  LIBS ${math_libs} base device cell_info parameter
  SOURCES write_orb_info_test.cpp ../write_orb_info.cpp ../output.cpp
)

AddTest(
  TARGET io_parse_args
  SOURCES parse_args_test.cpp ../parse_args.cpp
)

AddTest(
  TARGET io_bessel_basis_test
  LIBS ${math_libs} base device
  SOURCES bessel_basis_test.cpp ../bessel_basis.cpp
)

AddTest(
  TARGET io_output_log_test
  LIBS base ${math_libs} device
  SOURCES ../output_log.cpp outputlog_test.cpp ../../module_basis/module_pw/test/test_tool.cpp
)

AddTest(
  TARGET io_sparse_matrix_test
  LIBS base ${math_libs} device
  SOURCES sparse_matrix_test.cpp ../sparse_matrix.cpp
)

AddTest(
  TARGET io_file_reader_test
  LIBS base ${math_libs} device
  SOURCES file_reader_test.cpp ../file_reader.cpp
)

AddTest(
  TARGET io_csr_reader_test
  LIBS base ${math_libs} device
  SOURCES csr_reader_test.cpp ../csr_reader.cpp ../file_reader.cpp ../sparse_matrix.cpp
)

AddTest(
  TARGET read_rhog_test
  LIBS ${math_libs} base device planewave
  SOURCES read_rhog_test.cpp ../rhog_io.cpp ../binstream.cpp ../../module_basis/module_pw/test/test_tool.cpp
)

if(ENABLE_LCAO)
AddTest(
  TARGET to_qo_test
  LIBS base ${math_libs} device numerical_atomic_orbitals container orb parameter
  SOURCES 
  to_qo_test.cpp
  ../to_qo_kernel.cpp
  ../to_qo_mpi.cpp
  ../to_qo_structures.cpp
  ../../module_cell/atom_spec.cpp
  ../../module_cell/parallel_kpoints.cpp
  ../../module_cell/test/support/mock_unitcell.cpp
  ../../module_hamilt_lcao/hamilt_lcaodft/center2_orb.cpp
)
endif()

AddTest(
  TARGET read_wfc_pw_test
  LIBS base ${math_libs} device planewave
  SOURCES read_wfc_pw_test.cpp ../read_wfc_pw.cpp ../binstream.cpp ../../module_basis/module_pw/test/test_tool.cpp
)

add_test(NAME read_wfc_pw_test_parallel
      COMMAND mpirun -np 4 ./read_wfc_pw_test
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET numerical_basis_test
  LIBS base ${math_libs} device numerical_atomic_orbitals container orb parameter
  SOURCES numerical_basis_test.cpp ../numerical_basis_jyjy.cpp ../../module_hamilt_lcao/hamilt_lcaodft/center2_orb.cpp
)


AddTest(
  TARGET io_mulliken_test
  LIBS base ${math_libs} device
  SOURCES output_mulliken_test.cpp output_mulliken_mock.cpp ../output_mulliken.cpp
  ../../module_cell/cell_index.cpp
  ../../module_basis/module_ao/parallel_2d.cpp
  ../../module_basis/module_ao/parallel_orbitals.cpp
)

if(ENABLE_LCAO)
AddTest(
  TARGET read_wfc_lcao_test
  LIBS base ${math_libs} device
  SOURCES read_wfc_lcao_test.cpp ../read_wfc_lcao.cpp ../../module_basis/module_ao/parallel_2d.cpp
)

add_test(NAME read_wfc_lcao_test_parallel
      COMMAND mpirun -np 4 ./read_wfc_lcao_test
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

endif()

AddTest(
  TARGET cif_io_test
  LIBS base ${math_libs} device
  SOURCES cif_io_test.cpp ../cif_io.cpp
)

add_test(NAME cif_io_test_parallel
      COMMAND mpirun -np 4 ./cif_io_test
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)