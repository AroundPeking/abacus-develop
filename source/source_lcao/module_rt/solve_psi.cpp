#include "solve_psi.h"

#include <iostream>

#include "module_base/lapack_connector.h"
#include "module_base/scalapack_connector.h"

namespace module_tddft
{
void solve_psi_td(const Parallel_Orbitals* pv,
                    const int nband,
                    const int nlocal,
                    const std::complex<double>* U_operator_A,
                    const std::complex<double>* U_operator_B,
                    const std::complex<double>* psi_k_laststep,
                    std::complex<double>* psi_k,
                    const int print_matrix)
{
    std::complex<double>* tmp_b = new std::complex<double>[pv->nloc_wfc];
    //calculate B = U_operator_B * psi_k_laststep
    ScalapackConnector::gemm('N',
                        'N',
                        nlocal,
                        nband,
                        nlocal,
                        1.0,
                        U_operator_B,
                        1,
                        1,
                        pv->desc,
                        psi_k_laststep,
                        1,
                        1,
                        pv->desc_wfc,
                        0.0,
                        tmp_b,
                        1,
                        1,
                        pv->desc_wfc);
    //get ipiv
    int* ipiv = new int[pv->nloc];
    int info = 0;
    //solve Ax=B
    ScalapackConnector::gesv(nlocal,
                            nband,
                            U_operator_A,
                            1,
                            1,
                            pv->desc,
                            ipiv,
                            tmp_b,
                            1,
                            1,
                            pv->desc_wfc,
                            &info);

    //copy solution to psi_k
    BlasConnector::copy(pv->nloc_wfc, tmp_b, 1, psi_k, 1);
}
} // namespace module_tddft