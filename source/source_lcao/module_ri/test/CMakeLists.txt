remove_definitions(-D__MLALGO)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
AddTest(
  TARGET MODULE_RI_dm_mixing_test
  LIBS parameter base ${math_libs} device
  SOURCES dm_mixing_test.cpp ../Mix_DMk_2D.cpp ../Mix_Matrix.cpp
)
AddTest(
  TARGET MODULE_RI_ri_cv_io_test
  LIBS base ${math_libs} device parameter
  SOURCES ri_cv_io_test.cpp
)
AddTest(
  TARGET MODULE_RI_exx_compression_io_test
  LIBS base ${math_libs} device parameter
  SOURCES exx_compression_io_test.cpp
)
add_executable(MODULE_RI_ewald_tail_utils_test ewald_tail_utils_test.cpp)
install(TARGETS MODULE_RI_ewald_tail_utils_test DESTINATION ${CMAKE_BINARY_DIR}/tests)
add_test(
  NAME MODULE_RI_ewald_tail_utils_test
  COMMAND ./MODULE_RI_ewald_tail_utils_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_executable(MODULE_RI_gaussian_abfs_cutoff_test gaussian_abfs_cutoff_test.cpp)
install(TARGETS MODULE_RI_gaussian_abfs_cutoff_test DESTINATION ${CMAKE_BINARY_DIR}/tests)
add_test(
  NAME MODULE_RI_gaussian_abfs_cutoff_test
  COMMAND ./MODULE_RI_gaussian_abfs_cutoff_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_executable(MODULE_RI_librpa_stru_units_test librpa_stru_units_test.cpp)
install(TARGETS MODULE_RI_librpa_stru_units_test DESTINATION ${CMAKE_BINARY_DIR}/tests)
add_test(
  NAME MODULE_RI_librpa_stru_units_test
  COMMAND ./MODULE_RI_librpa_stru_units_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if(ENABLE_MPI)
  add_executable(MODULE_RI_ewald_mpi_distribution_test ewald_mpi_distribution_test.cpp)
  target_link_libraries(MODULE_RI_ewald_mpi_distribution_test MPI::MPI_CXX)
  install(TARGETS MODULE_RI_ewald_mpi_distribution_test DESTINATION ${CMAKE_BINARY_DIR}/tests)
  add_test(
    NAME MODULE_RI_ewald_mpi_distribution_test_parallel
    COMMAND mpirun -np 4 ./MODULE_RI_ewald_mpi_distribution_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
