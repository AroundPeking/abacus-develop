//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_H
#define EWALD_VQ_H

#include <RI/global/Tensor.h>
#include <mpi.h>

#include <array>
#include <map>

#include "LRI_CV.h"
#include "gaussian_abfs.h"
#include "module_base/element_basis_index.h"
#include "module_cell/klist.h"
#include "module_hamilt_general/module_xc/exx_info.h"

template <typename Tdata>
class Ewald_Vq
{
  private:
    using TA = int;
    using Tcell = int;
    static constexpr std::size_t Ndim = 3;
    using TC = std::array<Tcell, Ndim>;
    using TAC = std::pair<TA, TC>;

    using TK = std::array<int, 1>;
    using TAK = std::pair<TA, TK>;

  public:
    Ewald_Vq(const Exx_Info::Exx_Info_RI& info_in, const Exx_Info::Exx_Info_Ewald& info_ewald_in)
        : info(info_in), info_ewald(info_ewald_in)
    {
    }

    void init(const MPI_Comm& mpi_comm_in,
              std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos_in,
              std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_in,
              const K_Vectors* kv_in);

    void init_ions(
        const std::array<Tcell, Ndim>& period_Vs,
        const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>& list_As_Vs);

    double get_singular_chi();

    inline std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> cal_Vq(
        const double& chi,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in); // return Vq [0, Nk)
    inline std::map<TA, std::map<TAK, std::array<RI::Tensor<std::complex<double>>, Ndim>>> cal_dVq(
        std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>>& dVs_in); // return Vq [0, Nk)

    inline std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs(const double& chi,
                                                                 std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in);
    inline std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>> cal_dVs(
        std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>>& dVs_in);

  private:
    const Exx_Info::Exx_Info_RI& info;
    const Exx_Info::Exx_Info_Ewald& info_ewald;
    LRI_CV<Tdata> cv;
    Gaussian_Abfs gaussian_abfs;
    const K_Vectors* p_kv;
    std::vector<ModuleBase::Vector3<double>> kvec_c;
    std::vector<double> wk;
    MPI_Comm mpi_comm;
    ORB_gaunt_table MGT;
    std::vector<int> nmp;
    const double ewald_lambda = 1.0;

    std::vector<std::vector<std::vector<double>>> multipole;
    ModuleBase::Element_Basis_Index::IndexLNM index_abfs;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_lcaos;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs_ccp;

    std::vector<double> lcaos_rcut;
    std::vector<double> g_lcaos_rcut;

    const int nspin0 = std::map<int, int>{
        {1, 1},
        {2, 2},
        {4, 1}
    }.at(GlobalV::NSPIN);
    int nks0;
    std::vector<TA> atoms_vec;
    std::set<TA> atoms;

    /*
  MPI distribute
    distribute_atoms_periods:
      list_A0 / list_A1 : {ia0, {ia1, R}}
      list_A0_k / list_A1_k : {ia0, {ia1, ik}} ; range -> [-Nk/2, Nk/2)
    distribute_atoms:
      list_A0_pair_R / list_A1_pair_R : {ia0, ia1} for R
      list_A0_pair_k / list_A1_pair_k : {ia0, ia1} for k ; range -> [-Nk/2, Nk/2)
  */
    std::vector<TA> list_A0;
    std::vector<TAC> list_A1;
    std::vector<TA> list_A0_k;
    std::vector<TAK> list_A1_k;
    std::vector<TA> list_A0_pair_R;
    std::vector<TAC> list_A1_pair_R;
    std::vector<TA> list_A0_pair_k;
    std::vector<TAK> list_A1_pair_k;

  private:
    inline std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs_gauss(const std::vector<TA>& list_A0,
                                                                       const std::vector<TAC>& list_A1);
    inline std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>> cal_dVs_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1);

    inline std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs_minus_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in);
    inline std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>> cal_dVs_minus_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>>& dVs_in);
    template <typename Tresult>
    std::map<TA, std::map<TAC, Tresult>> set_Vs_dVs_minus_gauss(const std::vector<TA>& list_A0,
                                                                const std::vector<TAC>& list_A1,
                                                                std::map<TA, std::map<TAC, Tresult>>& Vs_dVs_in,
                                                                std::map<TA, std::map<TAC, Tresult>>& Vs_dVs_gauss_in);

    template <typename Tresult>
    using T_func_DPget_Vq_dVq
        = std::function<Tresult(const int& lp_max,
                                const int& lq_max, // Maximum L for which to calculate interaction.
                                const size_t& ik,
                                const ModuleBase::Vector3<double>& tau)>;
    inline std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> cal_Vq_gauss(
        const std::vector<TA>& list_A0_k,
        const std::vector<TAK>& list_A1_k,
        const double& chi,
        const int& shift_for_mpi); // return Vq [-Nk/2, Nk/2)
    inline std::map<TA, std::map<TAK, std::array<RI::Tensor<std::complex<double>>, Ndim>>> cal_dVq_gauss(
        const std::vector<TA>& list_A0_k,
        const std::vector<TAK>& list_A1_k,
        const int& shift_for_mpi); // return dVq [-Nk/2, Nk/2)
    template <typename Tresult>
    std::map<TA, std::map<TAK, Tresult>> set_Vq_dVq_gauss(const std::vector<TA>& list_A0_k,
                                                          const std::vector<TAK>& list_A1_k,
                                                          const int& shift_for_mpi,
                                                          const T_func_DPget_Vq_dVq<Tresult>& func_DPget_Vq_dVq);

    inline std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> cal_Vq_minus_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss); // return Vq [0, Nk)
    inline std::map<TA, std::map<TAK, std::array<RI::Tensor<std::complex<double>>, Ndim>>> cal_dVq_minus_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>>& dVs_minus_gauss); // return Vq [0, Nk)
    template <typename Tout, typename Tin>
    std::map<TA, std::map<TAK, Tout>> set_Vq_dVq_minus_gauss(const std::vector<TA>& list_A0,
                                                             const std::vector<TAC>& list_A1,
                                                             std::map<TA, std::map<TAC, Tin>>& Vs_dVs_minus_gauss);

    template <typename Tout, typename Tin>
    using T_func_DPcal_Vq_dVq_minus_gauss
        = std::function<std::map<TA, std::map<TAK, Tout>>(std::map<TA, std::map<TAC, Tin>>& Vs_dVs_minus_gauss)>;
    template <typename Tout>
    using T_func_DPcal_Vq_dVq_gauss = std::function<std::map<TA, std::map<TAK, Tout>>(const int& shift_for_mpi)>;
    template <typename Tout, typename Tin>
    std::map<TA, std::map<TAK, Tout>> set_Vq_dVq(
        const std::vector<TA>& list_A0_pair_k,
        const std::vector<TAK>& list_A1_pair_k,
        std::map<TA, std::map<TAC, Tin>>& Vs_dVs_minus_gauss_in,
        const T_func_DPcal_Vq_dVq_minus_gauss<Tout, Tin>& func_cal_Vq_dVq_minus_gauss,
        const T_func_DPcal_Vq_dVq_gauss<Tout>& func_cal_Vq_dVq_gauss); // return Vq [0, Nk)

    template <typename Tout, typename Tin>
    std::map<TA, std::map<TAC, Tout>> set_Vs_dVs(const std::vector<TA>& list_A0_pair_R,
                                                 const std::vector<TAC>& list_A1_pair_R,
                                                 std::map<TA, std::map<TAK, Tin>>& Vq);

    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> init_gauss(
        std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in);

    inline double get_Rcut_max(const int it0, const int it1);
    inline bool check_cell(const TC& cell);
};
#include "ewald_Vq.hpp"

#endif