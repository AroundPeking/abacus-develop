#ifndef TD_SOLVE_H
#define TD_SOLVE_H

#include "module_basis/module_ao/parallel_orbitals.h"
#include <complex>

namespace module_tddft
{
#ifdef __MPI
//solve Ax(t+dt) = Bx(t)
void solve_psi_td(const Parallel_Orbitals* pv,
                    const int nband,
                    const int nlocal,
                    const std::complex<double>* U_operator_A,
                    const std::complex<double>* U_operator_B,
                    const std::complex<double>* psi_k_laststep,
                    std::complex<double>* psi_k,
                    const int print_matrix);

#endif
} // namespace module_tddft

#endif // TD_SOLVE_H