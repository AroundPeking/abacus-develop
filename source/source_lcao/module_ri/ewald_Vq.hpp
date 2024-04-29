//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_HPP
#define EWALD_VQ_HPP

#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Global_Func-1.h>

// #include <chrono>
#include <cmath>

#include "RI_Util.h"
#include "conv_coulomb_pot_k-template.h"
#include "conv_coulomb_pot_k.h"
#include "exx_abfs-abfs_index.h"
#include "exx_abfs-construct_orbs.h"
#include "gaussian_abfs.h"
#include "module_base/element_basis_index.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "singular_value.h"

template <typename Tdata>
void Ewald_Vq<Tdata>::init(const MPI_Comm& mpi_comm_in,
                           std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos_in,
                           std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_in,
                           const K_Vectors* kv_in)
{
    ModuleBase::TITLE("Ewald_Vq", "init");
    ModuleBase::timer::tick("Ewald_Vq", "init");

    this->mpi_comm = mpi_comm_in;
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
                          this->info.ccp_rmesh_times,
                          false);
    this->multipole = Exx_Abfs::Construct_Orbs::get_multipole(abfs_in);

    const ModuleBase::Element_Basis_Index::Range range_abfs = Exx_Abfs::Abfs_Index::construct_range(abfs_in);
    this->index_abfs = ModuleBase::Element_Basis_Index::construct_index(range_abfs);

    this->MGT.init_Gaunt_CH(GlobalC::exx_info.info_ri.abfs_Lmax);
    this->MGT.init_Gaunt(GlobalC::exx_info.info_ri.abfs_Lmax);

    std::vector<int> values(Natoms);
    std::iota(values.begin(), values.end(), 0);
    this->atoms.insert(values.begin(), values.end());

    ModuleBase::timer::tick("Ewald_Vq", "init");
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::set_Vs(const std::vector<TA>& list_A0,
                             const std::vector<TAC>& list_A1,
                             const std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in)
    -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "init_Vs");
    ModuleBase::timer::tick("Ewald_Vq", "init_Vs");

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_gauss = this->cal_Vs_gauss(list_A0, list_A1);
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_minus_gauss_out
        = this->cal_Vs_minus_gauss(list_A0, list_A1, Vs_in, Vs_gauss);

    // MPI: {ia0, {ia1, R}} to {ia0, ia1}
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_minus_gauss
        = RI::Communicate_Tensors_Map_Judge comm_map2_first(this->mpi_comm,
                                                            Vs_minus_gauss_out,
                                                            this->atoms,
                                                            this->atoms);

    ModuleBase::timer::tick("Ewald_Vq", "init_Vs");
    return Vs_minus_gauss;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs_gauss(const std::vector<TA>& list_A0, const std::vector<TAC>& list_A1)
    -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_gauss");

    std::map<std::string, bool> flags = {
        {"writable_Vws", false}
    };

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_gauss = this->cv.cal_Vs(list_A0, list_A1, flags);

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_gauss");
    return Vs_gauss;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs_minus_gauss(const std::vector<TA>& list_A0,
                                         const std::vector<TAC>& list_A1,
                                         const std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_in,
                                         const std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_gauss_in)
    -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs_minus_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_minus_gauss");

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_minus_gauss;

#pragma omp parallel
    for (size_t i0 = 0; i0 < list_A0.size(); ++i0)
    {
#pragma omp for schedule(dynamic) nowait
        for (size_t i1 = 0; i1 < list_A1.size(); ++i1)
        {
            const TA iat0 = list_A0[i0];
            const int it0 = GlobalC::ucell.iat2it[iat0];
            const TA iat1 = list_A1[i1].first;
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
                                    const size_t index0 = this->index_abfs[it0][l0][n0][m0];
                                    const size_t index1 = this->index_abfs[it1][l1][n1][m1];
                                    data(index0, index1)
                                        = Vs_in[list_A0[i0]][list_A0[i1]](index0, index1)
                                          - pA * pB * Vs_gauss_in[list_A0[i0]][list_A0[i1]](index0, index1);
                                }
                            }
                        }
                    }
                }
            }
#pragma omp critical(Ewald_Vq_cal_Vs_minus_gauss)
            Vs_minus_gauss[list_A0[i0]][list_A0[i1]] = data;
        }
    }
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_minus_gauss");
    return Vs_minus_gauss;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::set_Vq(const std::vector<TA>& list_A0,
                             const std::vector<TAC>& list_A1,
                             const std::vector<TA>& list_A0_k,
                             const std::vector<TAK>& list_A1_k,
                             const double& chi,
                             const std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss)
    -> std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "set_Vq");
    ModuleBase::timer::tick("Ewald_Vq", "set_Vq");

    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> Vq;
    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> Vq_minus_gauss
        = this->cal_Vq_minus_gauss(list_A0, list_A1, Vs_minus_gauss);
    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> Vq_gauss = this->cal_Vq_gauss(list_A0_k, list_A1_k, chi);

#pragma omp parallel
    for (size_t i0 = 0; i0 < list_A0_k.size(); ++i0)
    {
#pragma omp for schedule(dynamic) nowait
        for (size_t i1 = 0; i1 < list_A1_k.size(); ++i1)
        {
            const TA iat0 = list_A0_k[i0];
            const int it0 = GlobalC::ucell.iat2it[iat0];
            const TA iat1 = list_A1_k[i1].first;
            const int it1 = GlobalC::ucell.iat2it[iat1];

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
                                const size_t index0 = this->index_abfs[it0][l0][n0][m0];
                                const size_t lm0 = l0 * l0 + m0;
                                for (size_t m1 = 0; m1 != 2 * l1 + 1; ++m1)
                                {
                                    const size_t index1 = this->index_abfs[it1][l1][n1][m1];
                                    const size_t lm1 = l1 * l1 + m1;
                                    data(index0, index1) = Vq_minus_gauss[list_A0_k[i0]][list_A1_k[i1]](index0, index1)
                                                           + pA * pB * Vq_gauss[list_A0_k[i0]][list_A1_k[i1]](lm0, lm1);
                                }
                            }
                        }
                    }
                }
            }

#pragma omp critical(Ewald_Vq_set_Vq)
            Vq[list_A0_k[i0]][list_A1_k[i1]] = data;
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "set_Vq");
    return Vq;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq_gauss(const std::vector<TA>& list_A0_k,
                                   const std::vector<TAK>& list_A1_k,
                                   const double& chi) -> std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vq_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_gauss");

    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> Vq_gauss_out;

    std::vector<ModuleBase::Vector3<double>> Gvec;
    Gvec.resize(3);
    Gvec[0].x = GlobalC::ucell.G.e11;
    Gvec[0].y = GlobalC::ucell.G.e12;
    Gvec[0].z = GlobalC::ucell.G.e13;

    Gvec[1].x = GlobalC::ucell.G.e21;
    Gvec[1].y = GlobalC::ucell.G.e22;
    Gvec[1].z = GlobalC::ucell.G.e23;

    Gvec[2].x = GlobalC::ucell.G.e31;
    Gvec[2].y = GlobalC::ucell.G.e32;
    Gvec[2].z = GlobalC::ucell.G.e33;

#pragma omp parallel
    for (size_t i0 = 0; i0 < list_A0_k.size(); ++i0)
    {
#pragma omp for schedule(dynamic) nowait
        for (size_t i1 = 0; i1 < list_A1_k.size(); ++i1)
        {
            const TA iat0 = list_A0_k[i0];
            const int it0 = GlobalC::ucell.iat2it[iat0];
            const int ia0 = GlobalC::ucell.iat2ia[iat0];
            const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];

            const TA iat1 = list_A1_k[i1].first;
            const int it1 = GlobalC::ucell.iat2it[iat1];
            const int ia1 = GlobalC::ucell.iat2ia[iat1];
            const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];
            const int ik = list_A1_k[i1].second[0];

            const ModuleBase::Vector3<double> tau = tau0 - tau1;

            RI::Tensor<std::complex<double>> data = this->gaussian_abfs.get_Vq(this->g_abfs_ccp[it0].size() - 1,
                                                                               this->g_abfs[it1].size() - 1,
                                                                               this->p_kv->kvec_c[ik],
                                                                               Gvec,
                                                                               chi,
                                                                               this->info_ewald.ewald_lambda,
                                                                               tau,
                                                                               this->MGT);

#pragma omp critical(Ewald_Vq_cal_Vq_gauss)
            Vq_gauss_out[list_A0_k[i0]][list_A1_k[i1]] = data;
        }
    }

    // MPI: {ia0, {ia1, k}} to {ia0, ia1}
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vq_gauss
        = RI::Communicate_Tensors_Map_Judge comm_map2_first(this->mpi_comm,
                                                            Vq_gauss_out,
                                                            this->atoms,
                                                            this->atoms);

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_gauss");
    return Vq_gauss
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq_minus_gauss(const std::vector<TA>& list_A0,
                                         const std::vector<TAC>& list_A1,
                                         std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_minus_gauss)
    -> std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vq_minus_gauss");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_minus_gauss");

    std::map<TA, std::map<TAK, RI::Tensor<std::complex<double>>>> datas;

    // auto start = std::chrono::system_clock::now();

    for (size_t ik = 0; ik != this->nks0; ++ik)
    {
#pragma omp parallel
        for (size_t i0 = 0; i0 < list_A0.size(); ++i0)
        {
#pragma omp for schedule(dynamic) nowait
            for (size_t i1 = 0; i1 < list_A1.size(); ++i1)
            {
                const TA iat0 = list_A0[i0];
                const TA iat1 = list_A1[i1].first;
                const TC& cell1 = list_A1[i1].second;
                std::complex<double> phase = std::exp(
                    ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                    * (this->p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1) * GlobalC::ucell.latvec)));

                RI::Tensor<std::complex<double>> Vs_tmp
                    = RI::Global_Func::convert<std::complex<double>>(Vs_minus_gauss[iat0][i0_ptr->second[i1]]) * phase;

#pragma omp critical(Ewald_Vq_cal_Vq_minus_gauss)
                {
                    const TAK index = std::make_pair(iat1, std::array<int, 1>{ik});
                    if (datas[iat0][index].empty())
                        datas[iat0][index] = Vs_tmp;
                    else
                        datas[iat0][index] = datas[iat0][index] + Vs_tmp;
                }
            }
        }
    }

    // auto end = std::chrono::system_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "cal_Vq_minus_gauss Time: "
    //           << double(duration.count()) * std::chrono::microseconds::period::num
    //                  / std::chrono::microseconds::period::den
    //           << " s" << std::endl;

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_minus_gauss");
    return datas;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs(const std::vector<TA>& list_A0,
                             const std::vector<TAC>& list_A1,
                             const std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>& Vq_in)
    -> std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs");

    const double SPIN_multiple = std::map<int, double>{
        {1, 0.5},
        {2, 1  },
        {4, 1  }
    }.at(GlobalV::NSPIN);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_minus_gauss = this->cal_Vs_minus_gauss();

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> datas;

    for (size_t ik = 0; ik != this->nks0; ++ik)
    {
        std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>> Vq
            = this->cal_Vq(this->p_kv->kvec_c[ik], Gvec, chi, Vs_minus_gauss);

        // auto start = std::chrono::system_clock::now();
#pragma omp parallel
        for (auto i0_ptr = this->list_A_cut.begin(); i0_ptr != this->list_A_cut.end(); ++i0_ptr)
        {
#pragma omp for schedule(dynamic) nowait
            for (size_t i1 = 0; i1 < i0_ptr->second.size(); ++i1)
            {
                const TA iat0 = i0_ptr->first;
                const int it0 = GlobalC::ucell.iat2it[iat0];
                const int ia0 = GlobalC::ucell.iat2ia[iat0];

                const TA iat1 = i0_ptr->second[i1].first;
                const TC& cell1 = i0_ptr->second[i1].second;
                const int it1 = GlobalC::ucell.iat2it[iat1];
                const int ia1 = GlobalC::ucell.iat2ia[iat1];

                const std::complex<double> frac
                    = std::exp(-ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                               * (this->p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1) * GlobalC::ucell.latvec)))
                      * this->p_kv->wk[ik] * SPIN_multiple;

                RI::Tensor<Tdata> Vs_tmp = RI::Global_Func::convert<Tdata>(Vq[iat0][iat1] * frac);

#pragma omp critical(Ewald_Vq_cal_Vs)
                {
                    if (datas[iat0][i0_ptr->second[i1]].empty())
                        datas[iat0][i0_ptr->second[i1]] = Vs_tmp;
                    else
                        datas[iat0][i0_ptr->second[i1]] = datas[iat0][i0_ptr->second[i1]] + Vs_tmp;
                }
            }
        }

        // auto end = std::chrono::system_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "cal_Vs_minus_gauss Time: "
        //           << double(duration.count()) * std::chrono::microseconds::period::num
        //                  / std::chrono::microseconds::period::den
        //           << " s" << std::endl;
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs");
    return datas;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq(const std::vector<TA>& list_A0_k, const std::vector<TAK>& list_A1_k, const double& chi)
    -> std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>
{
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
