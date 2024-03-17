//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_HPP
#define EWALD_VQ_HPP

#include <RI/global/Global_Func-1.h>

#include <cmath>

#include "RI_Util.h"
#include "conv_coulomb_pot_k-template.h"
#include "conv_coulomb_pot_k.h"
#include "exx_abfs-abfs_index.h"
#include "exx_abfs-construct_orbs.h"
#include "module_base/element_basis_index.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "singular_value.h"

template <typename Tdata>
void Ewald_Vq<Tdata>::init(std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos_in,
                           std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_in,
                           const K_Vectors* kv_in)
{
    ModuleBase::TITLE("Ewald_Vq", "init");
    ModuleBase::timer::tick("Ewald_Vq", "init");

    this->p_kv = kv_in;
    this->nks0 = this->p_kv->nkstot_full / this->nspin0;

    this->g_lcaos = this->init_gauss(lcaos_in);
    this->g_abfs = this->init_gauss(abfs_in);

    auto get_ccp_parameter = [this]() -> std::map<std::string, double> {
        switch (this->info.ccp_type)
        {
        case Conv_Coulomb_Pot_K::Ccp_Type::Ccp:
            return {};
        case Conv_Coulomb_Pot_K::Ccp_Type::Ccp_Cam:
            return {
                {"hse_omega", this->info.hse_omega},
                {"cam_alpha", this->info.cam_alpha},
                {"cam_beta",  this->info.cam_beta }
            };
        default:
            throw std::domain_error(std::string(__FILE__) + " line " + std::to_string(__LINE__));
            break;
        }
    };
    this->g_abfs_ccp = Conv_Coulomb_Pot_K::cal_orbs_ccp(this->g_abfs,
                                                        this->info.ccp_type,
                                                        get_ccp_parameter(),
                                                        this->info.ccp_rmesh_times,
                                                        this->p_kv->nkstot_full);
    this->cv.set_orbitals(this->g_lcaos,
                          this->g_abfs,
                          this->g_abfs_ccp,
                          this->info.kmesh_times,
                          this->info.ccp_rmesh_times);
    this->multipole = Exx_Abfs::Construct_Orbs::get_multipole(abfs_in);

    const ModuleBase::Element_Basis_Index::Range range_abfs = Exx_Abfs::Abfs_Index::construct_range(abfs_in);
    this->index_abfs = ModuleBase::Element_Basis_Index::construct_index(range_abfs);

    this->MGT.init_Gaunt(GlobalC::exx_info.info_ri.abfs_Lmax);

    ModuleBase::timer::tick("Ewald_Vq", "init");
}

template <typename Tdata>
void Ewald_Vq<Tdata>::init_atoms_from_Vs(const std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in)
{
    this->Vs = Vs_in;
    for (const auto& Vs_tmpA: this->Vs)
    {
        this->list_A0.push_back(Vs_tmpA.first);
        for (const auto& Vs_tmpB: Vs_tmpA.second)
            this->list_A1.push_back(Vs_tmpB.first);
    }
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs(std::vector<std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>>& Vq_in)
    -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs");

    const double SPIN_multiple = std::map<int, double>{
        {1, 0.5},
        {2, 1  },
        {4, 1  }
    }.at(GlobalV::NSPIN);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> datas;
    for (size_t ik = 0; ik != this->nks0; ++ik)
    {
        for (size_t i0 = 0; i0 < this->list_A0.size(); ++i0)
        {
            for (size_t i1 = 0; i1 < this->list_A1.size(); ++i1)
            {
                const TA iat0 = this->list_A0[i0];
                const int it0 = GlobalC::ucell.iat2it[iat0];
                const int ia0 = GlobalC::ucell.iat2ia[iat0];
                const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];

                const TA iat1 = this->list_A1[i1].first;
                const TC& cell1 = this->list_A1[i1].second;
                const int it1 = GlobalC::ucell.iat2it[iat1];
                const int ia1 = GlobalC::ucell.iat2ia[iat1];
                const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];

                const std::complex<double> frac
                    = std::exp(-ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                               * (this->p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1) * GlobalC::ucell.latvec)))
                      * this->p_kv->wk[ik] * SPIN_multiple;

                RI::Tensor<Tdata> Vs_tmp
                    = RI::Global_Func::convert<Tdata>(Vq_in[ik][this->list_A0[i0]][this->list_A1[i1]] * frac);
                if (datas[this->list_A0[i0]][this->list_A1[i1]].empty())
                    datas[this->list_A0[i0]][this->list_A1[i1]] = Vs_tmp;
                else
                    datas[this->list_A0[i0]][this->list_A1[i1]] = datas[this->list_A0[i0]][this->list_A1[i1]] + Vs_tmp;
            }
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs");
    return datas;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq(const ModulePW::PW_Basis_K* wfc_basis)
    -> std::vector<std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vq");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq");

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_minus_gauss = this->cal_Vs_minus_gauss();
    std::vector<std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>> Vq_minus_gauss
        = this->cal_Vq_minus_gauss(Vs_minus_gauss);
    std::vector<std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>> Vq;
    Vq.resize(this->nks0);

    double chi;
    switch (this->info_ewald.fq_type)
    {
    case Singular_Value::Fq_type::Type_0:
        chi = Singular_Value::cal_type_0(this->p_kv->kvec_c,
                                         this->info_ewald.ewald_qdiv,
                                         this->info_ewald.ewald_qdense,
                                         this->info_ewald.ewald_niter,
                                         this->info_ewald.ewald_eps,
                                         this->info_ewald.ewald_arate);
        break;
    case Singular_Value::Fq_type::Type_1:
        chi = Singular_Value::cal_type_1(this->p_kv->kvec_c,
                                         this->info_ewald.ewald_qdiv,
                                         wfc_basis,
                                         this->info_ewald.ewald_lambda,
                                         this->info_ewald.ewald_niter,
                                         this->info_ewald.ewald_eps);
        break;
    default:
        throw std::domain_error(std::string(__FILE__) + " line " + std::to_string(__LINE__));
        break;
    }

    for (size_t ik = 0; ik != this->nks0; ++ik)
    {
        for (size_t i0 = 0; i0 < this->list_A0.size(); ++i0)
        {
            for (size_t i1 = 0; i1 < this->list_A1.size(); ++i1)
            {
                const TA& iat0 = this->list_A0[i0];
                const int it0 = GlobalC::ucell.iat2it[iat0];
                const int ia0 = GlobalC::ucell.iat2ia[iat0];
                const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];

                const TA& iat1 = this->list_A1[i1].first;
                const TC& cell1 = this->list_A1[i1].second;
                const int it1 = GlobalC::ucell.iat2it[iat1];
                const int ia1 = GlobalC::ucell.iat2ia[iat1];
                const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];

                const ModuleBase::Vector3<double> tau = tau0 - tau1;
                RI::Tensor<std::complex<double>> Vq_gauss
                    = this->gaussian_abfs.get_Vq(GlobalC::exx_info.info_ri.abfs_Lmax,
                                                 GlobalC::exx_info.info_ri.abfs_Lmax,
                                                 this->p_kv->kvec_c[ik],
                                                 wfc_basis,
                                                 chi,
                                                 this->info_ewald.ewald_lambda,
                                                 tau,
                                                 this->MGT);

                const size_t size0 = this->index_abfs[it0].count_size;
                const size_t size1 = this->index_abfs[it1].count_size;
                RI::Tensor<std::complex<double>> data({size0, size1});

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
                                    const size_t i0 = this->index_abfs[it0][l0][n0][m0];
                                    const size_t lm0 = l0 * l0 + m0;
                                    for (size_t m1 = 0; m1 != 2 * l1 + 1; ++m1)
                                    {
                                        const size_t i1 = this->index_abfs[it1][l1][n1][m1];
                                        const size_t lm1 = l1 * l1 + m1;
                                        data(i0, i1) = Vq_minus_gauss[ik][this->list_A0[i0]][this->list_A1[i1]](i0, i1)
                                                       + pA * pB * Vq_gauss(lm0, lm1);
                                    }
                                }
                            }
                        }
                    }
                }

                Vq[ik][this->list_A0[i0]][this->list_A1[i1]] = data;
            }
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq");
    return Vq;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs_minus_gauss() -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs_minus_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_minus_gauss");

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_gauss = this->cal_Vs_gauss();
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_minus_gauss;

    for (size_t i0 = 0; i0 < this->list_A0.size(); ++i0)
    {
        for (size_t i1 = 0; i1 < this->list_A1.size(); ++i1)
        {
            const TA iat0 = this->list_A0[i0];
            const int it0 = GlobalC::ucell.iat2it[iat0];
            const TA iat1 = this->list_A1[i1].first;
            const int it1 = GlobalC::ucell.iat2it[iat1];

            const size_t size0 = this->index_abfs[it0].count_size;
            const size_t size1 = this->index_abfs[it1].count_size;
            RI::Tensor<Tdata> data({size0, size1});

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
                                    data(i0, i1) = this->Vs[this->list_A0[i0]][this->list_A1[i1]](i0, i1)
                                                   - pA * pB * Vs_gauss[this->list_A0[i0]][this->list_A1[i1]](i0, i1);
                                }
                            }
                        }
                    }
                }
            }

            Vs_minus_gauss[this->list_A0[i0]][this->list_A1[i1]] = data;
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_minus_gauss");
    return Vs_minus_gauss;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs_gauss() -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_gauss");

    std::map<std::string, bool> flags = {
        {"writable_Vws", false}
    };

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs = this->cv.cal_Vs(this->list_A0, this->list_A1, flags);

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_gauss");
    return Vs;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq_minus_gauss(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss)
    -> std::vector<std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vq_minus_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_minus_gauss");

    std::vector<std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>> datas;
    datas.resize(this->nks0);

    for (size_t ik = 0; ik != this->nks0; ++ik)
    {
        for (size_t i0 = 0; i0 < this->list_A0.size(); ++i0)
        {
            for (size_t i1 = 0; i1 < this->list_A1.size(); ++i1)
            {
                const TA iat0 = this->list_A0[i0];
                const TA iat1 = this->list_A1[i1].first;
                const TC& cell1 = this->list_A1[i1].second;
                std::complex<double> phase = std::exp(
                    ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                    * (this->p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1) * GlobalC::ucell.latvec)));
                if (datas[ik][this->list_A0[i0]][this->list_A1[i1]].empty())
                    datas[ik][this->list_A0[i0]][this->list_A1[i1]]
                        = RI::Global_Func::convert<std::complex<double>>(
                              Vs_minus_gauss[this->list_A0[i0]][this->list_A1[i1]])
                          * phase;
                else
                    datas[ik][this->list_A0[i0]][this->list_A1[i1]]
                        = datas[ik][this->list_A0[i0]][this->list_A1[i1]]
                          + RI::Global_Func::convert<std::complex<double>>(
                                Vs_minus_gauss[this->list_A0[i0]][this->list_A1[i1]])
                                * phase;
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
                gauss[T][L][N] = this->gaussian_abfs.Gauss(orb_in[T][L][N], this->info_ewald.ewald_lambda);
            }
        }
    }

    return gauss;
}

#endif
