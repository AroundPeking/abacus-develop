//=======================
// AUTHOR : jiyy
// DATE :   2023-12-08
//=======================

#ifndef EWALD_VQ_H
#define EWALD_VQ_H

#include <RI/global/Tensor.h>

#include <map>
#include <vector>

#include "module_base/abfs-vector3_order.h"
#include "module_base/complexmatrix.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_cell/klist.h"

template <typename Tdata>
class Ewald_Vq
{
  public:
    enum class Kernal_Type
    {
        Hf,
        Erfc, //  	"hse_omega"
    };

    enum class Auxiliary_Func
    {
        Type_1, // Phys. Rev. B, 75:205126, May 2007.
        Type_2, // Phys. Rev. B, 48:5058--5068, Aug 1993.
        Default = -1,
    };

    struct Ewald_Type
    {
        Kernal_Type ker_type;
        Auxiliary_Func aux_func;

        Ewald_Type(Kernal_Type ker_type_, Auxiliary_Func aux_func_ = Auxiliary_Func::Default)
            : ker_type(ker_type_), aux_func(aux_func_)
        {
            if (ker_type == Kernal_Type::Hf)
                assert(aux_func == Auxiliary_Func::Type_1 || aux_func == Auxiliary_Func::Type_2)
        }
    };
    Ewald_Type ewald_type;

  private:
    using TA = int;
    using TC = std::array<int, 3>;
    using TAC = std::pair<TA, TC>;

  public:
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs_ewald(
        const K_Vectors* kv,
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        const std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>>& Vq,
        const double& ccp_rmesh_times);

    std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>> cal_Vq1(
        const Ewald_Type& ewald_type,
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
        const K_Vectors* kv,
        const ModulePW::PW_Basis_K* wfc_basis,
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        const std::map<std::string, double>& parameter);

    std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>> cal_Vq2(
        const K_Vectors* kv,
        const std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs);

  private:
    const int nspin0 = std::map<int, int>{
        {1, 1},
        {2, 2},
        {4, 1}
    }.at(GlobalV::NSPIN);
    const double SPIN_multiple = std::map<int, double>{
        {1, 0.5},
        {2, 1  },
        {4, 1  }
    }.at(GlobalV::NSPIN);

  private:
    std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>,
              std::vector<std::vector<ModuleBase::ComplexMatrix>>>
        get_orb_q(const K_Vectors* kv,
                  const ModulePW::PW_Basis_K* wfc_basis,
                  const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
                  const double& gk_ecut);
    std::vector<ModuleBase::ComplexMatrix> produce_local_basis_in_pw(
        const int& ik,
        const std::vector<ModuleBase::Vector3<double>>& gk,
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
        const ModuleBase::realArray& table_local);

    static std::vector<int> get_npwk(const K_Vectors* kv, const ModulePW::PW_Basis_K* wfc_basis, const double& gk_ecut);
    static std::vector<std::vector<int>> get_igl2isz_k(const std::vector<int>& npwk,
                                                       const ModulePW::PW_Basis_K* wfc_basis);
    static std::vector<std::vector<ModuleBase::Vector3<double>>> get_gcar(const std::vector<int>& npwk,
                                                                          const ModulePW::PW_Basis_K* wfc_basis);
    static std::vector<double> cal_hf_kernel(const std::vector<ModuleBase::Vector3<double>>& gk);
    static std::vector<double> cal_erfc_kernel(const std::vector<ModuleBase::Vector3<double>>& gk, const double& omega);
    static double Iter_Integral(const T_cal_fq<double>& func_cal_fq,
                                const TC& nq_arr,
                                const int& niter,
                                const double& eps,
                                const int& a_rate);
    double solve_chi(const std::vector<ModuleBase::Vector3<double>>& gk,
                     const T_cal_fq<double>& func_cal_fq,
                     const TC& nq_arr,
                     const int& niter,
                     const double& eps,
                     const int& a_rate);
    double solve_chi(const std::vector<ModuleBase::Vector3<double>>& gk,
                     const T_cal_fq<double>& func_cal_fq,
                     const double& fq_int);

    // TODO: Here, fq now only works on 3D and 2D systems
    // TODO: lower dimension please see PHYSICAL REVIEW B 87, 165122 (2013)

    // qdiv=2 i.e. q^{-2} for 3D;
    // qdiv=1 i.e. q^{-1} for 2D.
    using T_cal_fq_type_1
        = std::function<double(const std::vector<ModuleBase::Vector3<double>>& gk, const double& qdiv)>;
    static double fq_type_1(const ModuleBase::Vector3<double>& qvec,
                            const int& qdiv,
                            std::vector<ModuleBase::Vector3<double>>& avec,
                            std::vector<ModuleBase::Vector3<double>>& bvec);
    double cal_type_1(const std::vector<ModuleBase::Vector3<double>>& gk,
                      const int& qdiv,
                      const double& qdense,
                      const int& niter,
                      const double& eps,
                      const int& a_rate);
    // gamma: chosen as the radius of sphere which has the same volume as the Brillouin zone.
    using T_cal_fq_type_2 = std::function<double(const std::vector<ModuleBase::Vector3<double>>& gk,
                                                 const double& qdiv,
                                                 const ModulePW::PW_Basis_K* wfc_basis,
                                                 const double& lambda)>;
    static double fq_type_2(const ModuleBase::Vector3<double>& qvec,
                            const int& qdiv,
                            const ModulePW::PW_Basis_K* wfc_basis,
                            const double& lambda);
    double cal_type_2(const std::vector<ModuleBase::Vector3<double>>& gk,
                      const int& qdiv,
                      const ModulePW::PW_Basis_K* wfc_basis,
                      const double& lambda)
};

#include "ewald_Vq.hpp"

#endif