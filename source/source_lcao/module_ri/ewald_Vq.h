//=======================
// AUTHOR : jiyy
// DATE :   2023-12-08
//=======================

#ifndef EWALD_VQ_H
#define EWALD_VQ_H

#include <RI/global/Tensor.h>

#include <array>
#include <map>
#include <vector>

#include "auxiliary_func.h"
#include "module_base/abfs-vector3_order.h"
#include "module_base/complexmatrix.h"
#include "module_base/global_variable.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_cell/klist.h"

template <typename Tdata>
class Ewald_Vq
{
  private:
    using TA = int;
    using TC = std::array<int, 3>;
    using TAC = std::pair<TA, TC>;
    using T_kernal_func = std::function<std::vector<double>(std::vector<ModuleBase::Vector3<double>>& gk)>;

  public:
    /*-------------------------------------------
    cal_Vs_ewald:
      in ->
        Vs_sr: non-periodic R
        Vq: full Vq
      out ->
        Vs_full: cam_alpha * Vs_full based on periodic R + cam_beta * Vs_sr based on non-periodic R
    -------------------------------------------*/
    void cal_Vs_ewald(const K_Vectors* kv,
                      const UnitCell& ucell,
                      std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs,
                      std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>>& Vq,
                      const std::vector<TA>& list_A0,
                      const std::vector<TAC>& list_A1,
                      const double& cam_alpha,
                      const double& cam_beta);

    //\sum_G P*(q-G)v(q-G)P(q-G)\exp(-i(q-G)\tau)
    std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>> cal_Vq_q(
        const Auxiliary_Func::Kernal_Type& ker_type,
        const Auxiliary_Func::Fq_type& fq_type,
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
        const K_Vectors* kv,
        const UnitCell& ucell,
        const ModulePW::PW_Basis_K* wfc_basis,
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        const std::map<std::string, double>& parameter);

    //\sum_R V(R)\exp(iqR)
    std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>> cal_Vq_R(
        const K_Vectors* kv,
        const UnitCell& ucell,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs);

  private:
    const int nspin0 = std::map<int, int>{
        {1, 1},
        {2, 2},
        {4, 1}
    }.at(GlobalV::NSPIN);

  private:
    std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>,
              std::vector<std::vector<ModuleBase::ComplexMatrix>>>
        get_orb_q(std::vector<ModuleBase::Vector3<double>>& kvec_c,
                  const ModulePW::PW_Basis_K* wfc_basis,
                  const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
                  const double& gk_ecut);
    std::vector<ModuleBase::ComplexMatrix> produce_local_basis_in_pw(
        const int& ik,
        const std::vector<ModuleBase::Vector3<double>>& gk,
        const double& tpiba,
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
        const ModuleBase::realArray& table_local);

    static std::vector<int> get_npwk(std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                     const ModulePW::PW_Basis_K* wfc_basis,
                                     const double& gk_ecut);
    static std::vector<std::vector<int>> get_igl2isz_k(const std::vector<int>& npwk,
                                                       const ModulePW::PW_Basis_K* wfc_basis);
    static std::vector<std::vector<ModuleBase::Vector3<double>>> get_gcar(const std::vector<int>& npwk,
                                                                          const ModulePW::PW_Basis_K* wfc_basis);
};

#include "ewald_Vq.hpp"

#endif