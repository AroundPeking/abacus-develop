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
    using TC = std::array<int, 3>;
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

    /*
    MPI distribute
      distribute_atoms_periods:
        list_A0 / list_A1 : {ia0, {ia1, R}}
        list_A0_k / list_A1_k : {ia0, {ia1, ik}} ; range -> [-Nk/2, Nk/2)
      distribute_atoms:
        list_A0_pair_R / list_A1_pair_R : {ia0, ia1} for R
        list_A0_pair_k / list_A1_pair_k : {ia0, ia1} for k ; range -> [-Nk/2, Nk/2)
    */
    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> cal_Vq(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        const std::vector<TA>& list_A0_k,
        const std::vector<TAK>& list_A1_k,
        const std::vector<TA>& list_A0_pair_R,
        const std::vector<TAC>& list_A1_pair_R,
        const std::vector<TA>& list_A0_pair_k,
        const std::vector<TAK>& list_A1_pair_k,
        const double& chi,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in); // return Vq [0, Nk)

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs(const std::vector<TA>& list_A0,
                                                          const std::vector<TAC>& list_A1,
                                                          const std::vector<TA>& list_A0_k,
                                                          const std::vector<TAK>& list_A1_k,
                                                          const std::vector<TA>& list_A0_pair_R,
                                                          const std::vector<TAC>& list_A1_pair_R,
                                                          const std::vector<TA>& list_A0_pair_k,
                                                          const std::vector<TAK>& list_A1_pair_k,
                                                          const double& chi,
                                                          std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in);

  private:
    const Exx_Info::Exx_Info_RI& info;
    const Exx_Info::Exx_Info_Ewald& info_ewald;
    LRI_CV<Tdata> cv;
    Gaussian_Abfs gaussian_abfs;
    const K_Vectors* p_kv;
    MPI_Comm mpi_comm;
    ORB_gaunt_table MGT;
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
    std::set<TA> atoms;

  private:
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> set_Vs(
        const std::vector<TA>& list_A0_pair_R,
        const std::vector<TAC>& list_A1_pair_R,
        std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> Vq);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs_minus_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs_gauss(const std::vector<TA>& list_A0,
                                                                const std::vector<TAC>& list_A1);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> set_Vs_minus_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_gauss_in);

    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> set_Vq(
        const std::vector<TA>& list_A0_k,
        const std::vector<TAK>& list_A1_k,
        const std::vector<TA>& list_A0_pair_R,
        const std::vector<TAC>& list_A1_pair_R,
        const std::vector<TA>& list_A0_pair_k,
        const std::vector<TAK>& list_A1_pair_k,
        const double& chi,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss_in); // return Vq [0, Nk)

    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> cal_Vq_gauss(
        const std::vector<TA>& list_A0_k,
        const std::vector<TAK>& list_A1_k,
        const double& chi,
        const int& shift_for_mpi); // return Vq [-Nk/2, Nk/2)

    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> cal_Vq_minus_gauss(
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss); // return Vq [0, Nk)

    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> init_gauss(
        std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in);

    inline double get_Rcut_max(const int it0, const int it1);
};
#include "ewald_Vq.hpp"

#endif