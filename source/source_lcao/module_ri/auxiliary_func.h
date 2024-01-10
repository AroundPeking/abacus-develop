//=======================
// AUTHOR : jiyy
// DATE :   2024-01-10
//=======================

#ifndef AUXILIARY_FUNC_H
#define AUXILIARY_FUNC_H

#include <array>
#include <vector>

#include "module_basis/module_pw/pw_basis_k.h"

class Auxiliary_Func
{
  private:
    using T_cal_fq_type_0
        = std::function<double(const std::vector<ModuleBase::Vector3<double>>& gk, const double& qdiv)>;
    using T_cal_fq_type_1 = std::function<double(const std::vector<ModuleBase::Vector3<double>>& gk,
                                                 const double& qdiv,
                                                 const ModulePW::PW_Basis_K* wfc_basis,
                                                 const double& lambda)>;

  private:
    static double Iter_Integral(const T_cal_fq_type_0& func_cal_fq,
                                const std::array<int, 3>& nq_arr,
                                const int& niter,
                                const double& eps,
                                const int& a_rate);
    double solve_chi(const std::vector<ModuleBase::Vector3<double>>& gk,
                     const T_cal_fq_type_0& func_cal_fq,
                     const std::array<int, 3>& nq_arr,
                     const int& niter,
                     const double& eps,
                     const int& a_rate);
    double solve_chi(const std::vector<ModuleBase::Vector3<double>>& gk,
                     const T_cal_fq_type_1& func_cal_fq,
                     const double& fq_int);

    // TODO: Here, fq now only works on 3D and 2D systems
    // TODO: lower dimension please see PHYSICAL REVIEW B 87, 165122 (2013)

    // qdiv=2 i.e. q^{-2} for 3D;
    // qdiv=1 i.e. q^{-1} for 2D.
    static double fq_type_0(const ModuleBase::Vector3<double>& qvec,
                            const int& qdiv,
                            std::vector<ModuleBase::Vector3<double>>& avec,
                            std::vector<ModuleBase::Vector3<double>>& bvec);
    // gamma: chosen as the radius of sphere which has the same volume as the Brillouin zone.
    static double fq_type_1(const ModuleBase::Vector3<double>& qvec,
                            const int& qdiv,
                            const ModulePW::PW_Basis_K* wfc_basis,
                            const double& lambda);

  public:
    static std::vector<double> cal_hf_kernel(const std::vector<ModuleBase::Vector3<double>>& gk);
    static std::vector<double> cal_erfc_kernel(const std::vector<ModuleBase::Vector3<double>>& gk, const double& omega);

    static double cal_type_0(const std::vector<ModuleBase::Vector3<double>>& gk,
                             const int& qdiv,
                             const double& qdense,
                             const int& niter,
                             const double& eps,
                             const int& a_rate);
    static double cal_type_1(const std::vector<ModuleBase::Vector3<double>>& gk,
                             const int& qdiv,
                             const ModulePW::PW_Basis_K* wfc_basis,
                             const double& lambda);
};

#endif