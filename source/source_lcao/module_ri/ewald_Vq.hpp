//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_HPP
#define EWALD_VQ_HPP

#include "module_base/element_basis_index.h"
#include "module_ri/conv_coulomb_pot_k-template.h"
#include "module_ri/conv_coulomb_pot_k.h"
#include "module_ri/exx_abfs-construct_orbs.h"

template <typename Tdata>
void Ewald_Vq<Tdata>::init(std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos_in,
                           std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_in,
                           ModuleBase::Element_Basis_Index::IndexLNM& index_abfs_in,
                           const K_Vectors& kv_in,
                           const double rmesh_times,
                           const double& gauss_gamma)
{
    ModuleBase::TITLE("Ewald_Vq", "init");
    ModuleBase::timer::tick("Ewald_Vq", "init");

    this->p_kv = &kv_in;
    this->gamma = gauss_gamma;
    this->g_lcaos = this->init_gauss(lcaos_in);
    this->g_abfs = this->init_gauss(abfs_in);

    this->g_abfs_ccp = Conv_Coulomb_Pot_K::cal_orbs_ccp(this->g_abfs,
                                                        Conv_Coulomb_Pot_K::Ccp_Type::Ccp,
                                                        {},
                                                        this->info.ccp_rmesh_times,
                                                        p_kv->nkstot_full);
    this->cv.set_orbitals(this->g_lcaos,
                          this->g_abfs,
                          this->g_abfs_ccp,
                          this->info.kmesh_times,
                          this->info.ccp_rmesh_times);
    this->multipole = Exx_Abfs::Construct_Orbs::get_multipole(abfs_in);

    this->index_abfs = index_abfs_in;

    ModuleBase::timer::tick("Ewald_Vq", "init");
}

template <typename Tdata>
void Ewald_Vq<Tdata>::cal_Vs(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs,
                             const std::map<std::string, bool>& flags)
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs");

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_gauss = this->cal_Vs_gauss(list_A0, list_A1, flags);

    std::map<TA, std::map<TAC, Tresult>> Datas;
    for (const auto& Vs_tmpA: Vs)
    {
        const TA& iat0 = Vs_tmpA.first;
        const int it0 = GlobalC::ucell.iat2it[iat0];
        for (const auto& Vs_tmpB: Vs_tmpA.second)
        {
            const TA& iat1 = Vs_tmpB.first.first;
            const int it1 = GlobalC::ucell.iat2it[iat1];
            // V(R) = V(R) - pA * pB * V(R)_gauss
            for (int LA = 0; LA != this->g_abfs_ccp[it0].size(); ++LA)
            {
                for (size_t NA = 0; NA != this->g_abfs_ccp[it0][LA].size(); ++NA)
                {
                    const double pA = this->multipole[it0][LA][NA];
                    for (size_t MA = 0; MA != 2 * LA + 1; ++MA)
                    {
                        for (int LB = 0; LB != this->g_abfs[it1].size(); ++LB)
                        {
                            for (size_t NB = 0; NB != this->g_abfs[it1][LB].size(); ++NB)
                            {
                                const double pB = this->multipole[it1][LB][NB];
                                for (size_t MB = 0; MB != 2 * LB + 1; ++MB)
                                {
                                    const size_t iA = this->index_abfs[it0][LA][NA][MA];
                                    const size_t iB = this->index_abfs[it1][LB][NB][MB];
                                    Vs[iat0][Vs_tmpB.first](iA, iB) -= pA * pB * Vs_gauss[iat0][Vs_tmpB.first](iA, iB);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs");
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs_gauss(const std::vector<TA>& list_A0,
                                   const std::vector<TAC>& list_A1,
                                   const std::map<std::string, bool>& flags)
    -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_gauss");

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs = this->cv.cal_Vs(list_A0, list_A1, flags);

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_gauss");
    return Vs;
}

template <typename Tdata>
std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> Ewald_Vq<Tdata>::init_gauss(
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in)
{
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> gauss;
    gauss.resize(orb_in.size());
    for (size_t T = 0; T != orb_in.size(); ++T)
    {
        gauss[T].resize(orb_in[T].size());
        for (size_t L = 0; L != orb_in[T].size(); ++L)
        {
            gauss[T][L].resize(orb_in[T][L].size());
            for (size_t N = 0; N != orb_in[T][L].size(); ++N)
            {
                gauss[T][L][N] = this->gaussian_abfs.Gauss(orb_in[T][L][N], this->gamma);
            }
        }
    }

    return gauss;
}

#endif
