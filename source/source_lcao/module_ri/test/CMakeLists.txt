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
  TARGET MODULE_RI_sternheimer_rpa_test
  LIBS ${math_libs}
  SOURCES sternheimer_rpa_test.cpp ../sternheimer_rpa.cpp
)
AddTest(
  TARGET MODULE_RI_sternheimer_fd_hamiltonian_test
  LIBS ${math_libs}
  SOURCES sternheimer_fd_hamiltonian_test.cpp ../sternheimer_fd_hamiltonian.cpp ../sternheimer_fd_nonlocal_projector.cpp
)
AddTest(
  TARGET MODULE_RI_sternheimer_fd_solver_test
  LIBS ${math_libs}
  SOURCES sternheimer_fd_solver_test.cpp ../sternheimer_fd_solver.cpp ../sternheimer_fd_hamiltonian.cpp ../sternheimer_fd_nonlocal_projector.cpp ../sternheimer_rpa.cpp
)
AddTest(
  TARGET MODULE_RI_sternheimer_dft_zero_order_test
  LIBS ${math_libs}
  SOURCES sternheimer_dft_zero_order_test.cpp ../sternheimer_dft_zero_order.cpp ../sternheimer_fd_solver.cpp ../sternheimer_fd_hamiltonian.cpp ../sternheimer_fd_nonlocal_projector.cpp ../sternheimer_rpa.cpp
)
AddTest(
  TARGET MODULE_RI_sternheimer_fd_nonlocal_projector_test
  LIBS ${math_libs}
  SOURCES sternheimer_fd_nonlocal_projector_test.cpp ../sternheimer_fd_nonlocal_projector.cpp
)
AddTest(
  TARGET MODULE_RI_sternheimer_abacus_fd_adapter_test
  LIBS parameter base ${math_libs} device container
  SOURCES sternheimer_abacus_fd_adapter_test.cpp ../sternheimer_abacus_fd_adapter.cpp ../sternheimer_fd_hamiltonian.cpp ../sternheimer_fd_nonlocal_projector.cpp
)
install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
