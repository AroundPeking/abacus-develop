//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_H
#define EWALD_VQ_H

#include <RI/global/Tensor.h>

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

  public:
    Ewald_Vq(const Exx_Info::Exx_Info_RI& info_in, const Exx_Info::Exx_Info_Ewald& info_ewald_in)
        : info(info_in), info_ewald(info_ewald_in)
    {
    }

    void init(std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos_in,
              std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_in,
              const K_Vectors* kv_in);

    void init_atoms_from_Vs(const std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in,
                            const std::vector<TA>& list_A0_in,
                            const std::vector<TAC>& list_A1_in);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs(const ModulePW::PW_Basis_K* wfc_basis);

    std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>> cal_Vq(const ModuleBase::Vector3<double>& qvec,
                                                                        const ModulePW::PW_Basis_K* wfc_basis,
                                                                        const double& chi);

  private:
    const Exx_Info::Exx_Info_RI& info;
    const Exx_Info::Exx_Info_Ewald& info_ewald;
    LRI_CV<Tdata> cv;
    Gaussian_Abfs gaussian_abfs;
    const K_Vectors* p_kv;
    ORB_gaunt_table MGT;

    std::vector<std::vector<std::vector<double>>> multipole;
    ModuleBase::Element_Basis_Index::IndexLNM index_abfs;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_lcaos;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs_ccp;

    const int nspin0 = std::map<int, int>{
        {1, 1},
        {2, 2},
        {4, 1}
    }.at(GlobalV::NSPIN);
    int nks0;

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs;
    std::vector<TA> list_A0;
    std::vector<TAC> list_A1;
    std::map<TA, std::vector<TAC>> list_A_cut;
    std::map<TA, std::vector<TA>> list_B;

  private:
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs_minus_gauss();

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> cal_Vs_gauss();

    std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>> cal_Vq_minus_gauss(
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss,
        const ModuleBase::Vector3<double>& qvec);

    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> init_gauss(
        std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in);
};
#include "ewald_Vq.hpp"

#endif