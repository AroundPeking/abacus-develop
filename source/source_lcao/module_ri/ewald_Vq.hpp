//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_HPP
#define EWALD_VQ_HPP

#include <RI/global/Global_Func-1.h>

#include <cmath>

#include "RI_Util.h"
#include "module_base/element_basis_index.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_ri/conv_coulomb_pot_k-template.h"
#include "module_ri/conv_coulomb_pot_k.h"
#include "module_ri/exx_abfs-construct_orbs.h"

template <typename Tdata>
void Ewald_Vq<Tdata>::init(std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos_in,
                           std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_in,
                           ModuleBase::Element_Basis_Index::IndexLNM& index_abfs_in,
                           const K_Vectors& kv_in,
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
auto Ewald_Vq<Tdata>::cal_Vs_minus_gauss(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in,
                                         const std::map<std::string, bool>& flags)
    -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs_minus_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_minus_gauss");

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_gauss = this->cal_Vs_gauss(list_A0, list_A1, flags);
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs;

    std::map<TA, std::map<TAC, Tresult>> Datas;
    for (const auto& Vs_tmpA: Vs)
    {
        const TA& iat0 = Vs_tmpA.first;
        const int it0 = GlobalC::ucell.iat2it[iat0];
        for (const auto& Vs_tmpB: Vs_tmpA.second)
        {
            const TA& iat1 = Vs_tmpB.first.first;
            const int it1 = GlobalC::ucell.iat2it[iat1];
            const size_t size0 = this->index_abfs[it0].count_size;
            const size_t size1 = this->index_abfs[it1].count_size;
            RI::Tensor<std::complex<double>> data({size0, size1});

            // V(R) = V(R) - pA * pB * V(R)_gauss
            for (int l0 = 0; l0 != this->g_abfs_ccp[it0].size(); ++l0)
            {
                for (int l1 = 0; l1 != this->g_abfs[it1].size(); ++l1)
                {
                    for (size_t n0 = 0; n0 != this->g_abfs_ccp[it0][l0].size(); ++n0)
                    {
                        const double pA = this->multipole[it0][l0][n0];
                        for (size_t n1 = 0; n1 != this->g_abfs[it1][l1].size(); ++n1)
                        {
                            const double pB = this->multipole[it1][l1][n1];
                            for (size_t m0 = 0; m0 != 2 * l0 + 1; ++m0)
                            {
                                for (size_t m1 = 0; m1 != 2 * l1 + 1; ++m1)
                                {
                                    const size_t i0 = this->index_abfs[it0][l0][n0][m0];
                                    const size_t i1 = this->index_abfs[it1][l1][n1][m1];
                                    data(i0, i1) = Vs_in[iat0][Vs_tmpB.first](i0, i1)
                                                   - pA * pB * Vs_gauss[iat0][Vs_tmpB.first](i0, i1);
                                }
                            }
                        }
                    }
                }
            }

            Vs[iat0][Vs_tmpB.first] = data;
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_minus_gauss");
    return Vs;
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
auto Ewald_Vq<Tdata>::cal_Vq_minus_gauss(const K_Vectors* kv,
                                         std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss)
    -> std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vq_minus_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_minus_gauss");

    const int nks0 = kv->nks / this->nspin0;
    std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>> datas;
    datas.resize(nks0);

    for (size_t ik = 0; ik != nks0; ++ik)
    {
        for (const auto& Vs_tmpA: Vs_minus_gauss)
        {
            const TA& iat0 = Vs_tmpA.first;
            for (const auto& Vs_tmpB: Vs_tmpA.second)
            {
                const TA& iat1 = Vs_tmpB.first.first;
                const TC& cell1 = Vs_tmpB.first.second;
                std::complex<double> phase
                    = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                               * (kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1) * GlobalC::ucell.latvec)));
                if (datas[ik][iat0][iat1].empty())
                    datas[ik][iat0][iat1]
                        = RI::Global_Func::convert<std::complex<double>>(Vs_minus_gauss[iat0][Vs_tmpB.first]) * phase;
                else
                    datas[ik][iat0][iat1]
                        = datas[ik][iat0][iat1]
                          + RI::Global_Func::convert<std::complex<double>>(Vs_minus_gauss[iat0][Vs_tmpB.first]) * phase;
            }
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_minus_gauss");
    return datas;
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
