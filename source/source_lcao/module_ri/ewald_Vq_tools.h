//=======================
// AUTHOR : jiyy
// DATE :   2024-01-21
//=======================

#ifndef EWALD_VQ_TOOLS_H
#define EWALD_VQ_TOOLS_H

#include <array>
#include <map>
#include <vector>

#include "module_base/complexmatrix.h"
#include "module_base/global_variable.h"
#include "module_base/realarray.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_basis/module_pw/pw_basis_k.h"

class Ewald_Vq_tools
{
  public:
    static std::vector<ModuleBase::ComplexMatrix> produce_local_basis_in_pw(
        const std::vector<ModuleBase::Vector3<double>>& gk,
        const double& tpiba,
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
        const ModuleBase::realArray& table_local);

    static std::vector<int> get_npwk(std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                     const ModulePW::PW_Basis_K* wfc_basis,
                                     const double& gk_ecut);
    static std::vector<std::vector<ModuleBase::Vector3<double>>> get_gcar(const std::vector<int>& npwk,
                                                                          const ModulePW::PW_Basis_K* wfc_basis);

  private:
    const int nspin0 = std::map<int, int>{
        {1, 1},
        {2, 2},
        {4, 1}
    }.at(GlobalV::NSPIN);

  private:
    static std::vector<std::vector<int>> get_igl2isz_k(const std::vector<int>& npwk,
                                                       const ModulePW::PW_Basis_K* wfc_basis);
};

#endif