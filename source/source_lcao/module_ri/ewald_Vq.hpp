//=======================
// AUTHOR : jiyy
// DATE :   2023-12-08
//=======================

#ifndef EWALD_VQ_HPP
#define EWALD_VQ_HPP

#include <omp.h>

#include <cmath>

#include "RI_Util.h"
#include "auxiliary_func.h"
#include "ewald_Vq.h"
#include "exx_abfs-construct_orbs.h"
#include "module_base/math_polyint.h"
#include "module_base/math_ylmreal.h"
#include "module_base/realarray.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_lcao/hamilt_lcaodft/wavefunc_in_pw.h"

template <typename Tdata>
void Ewald_Vq<Tdata>::cal_Vs_ewald(const K_Vectors* kv,
                                   const UnitCell& ucell,
                                   std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs,
                                   std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>>& Vq,
                                   const std::vector<TA>& list_A0,
                                   const std::vector<TAC>& list_A1,
                                   const double& cam_alpha,
                                   const double& cam_beta)
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vs_ewald");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_ewald");

    const TC period = RI_Util::get_Born_vonKarmen_period(*kv);
    const int nks0 = kv->nks / this->nspin0;
    const double SPIN_multiple = std::map<int, double>{
        {1, 0.5},
        {2, 1  },
        {4, 1  }
    }.at(GlobalV::NSPIN);
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> datas;

    for (size_t i0 = 0; i0 != list_A0.size(); ++i0)
    {
        const TA iat0 = list_A0[i0];
        for (size_t i1 = 0; i1 != list_A1.size(); ++i1)
        {
            const TAC pair_nonperiod = list_A1[i1];
            const TA iat1 = pair_nonperiod.first;
            const TC cell1 = pair_nonperiod.second;

            using namespace RI::Array_Operator;
            const TC cell1_period = cell1 % period;

            RI::Tensor<Tdata> Vs_tmp;
            for (size_t ik = 0; ik != nks0; ++ik)
            {
                const std::complex<double> frac
                    = cam_alpha
                      * std::exp(-ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                                 * (kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1_period) * ucell.latvec)))
                      * kv->wk[ik] * SPIN_multiple;
                RI::Tensor<Tdata> Vs_full_tmp = RI::Global_Func::convert<Tdata>(Vq[ik][iat0][iat1] * frac);

                if (Vs_tmp.empty())
                    Vs_tmp = Vs_full_tmp;
                else
                    Vs_tmp += Vs_full_tmp;
            }
            Vs[iat0][pair_nonperiod] = Vs[iat0][pair_nonperiod].empty()
                                           ? RI::Tensor<Tdata>(Vs_tmp.shape)
                                           : RI::Global_Func::convert<Tdata>(cam_beta) * Vs[iat0][pair_nonperiod];
            const TAC pair_period = std::make_pair(iat1, cell1_period);
            Vs[iat0][pair_period] = Vs[iat0][pair_period].empty() ? Vs_tmp : Vs[iat0][pair_period] + Vs_tmp;
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vs_ewald");
}

// Zc
template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq_q(const Auxiliary_Func::Kernal_Type& ker_type,
                               const Auxiliary_Func::Fq_type& fq_type,
                               const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
                               const K_Vectors* kv,
                               const UnitCell& ucell,
                               const ModulePW::PW_Basis_K* wfc_basis,
                               const std::vector<TA>& list_A0,
                               const std::vector<TAC>& list_A1,
                               const std::map<std::string, double>& parameter)
    -> std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vq_q");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_q");

    const int nks0 = kv->nks / this->nspin0;
    std::vector<ModuleBase::Vector3<double>> new_kvec(nks0);
    std::transform(kv->kvec_c.begin(),
                   kv->kvec_c.begin() + nks0,
                   new_kvec.begin(),
                   [](const ModuleBase::Vector3<double>& vec) { return vec; });
    double chi = 0;
    T_kernal_func cal_kernal;
    switch (ker_type)
    {
    case Auxiliary_Func::Kernal_Type::Hf:
        switch (fq_type)
        {
        case Auxiliary_Func::Fq_type::Type_0:
            chi = Auxiliary_Func::cal_type_0(new_kvec,
                                             static_cast<int>(parameter.at("ewald_qdiv")),
                                             parameter.at("ewald_qdense"),
                                             static_cast<int>(parameter.at("ewald_niter")),
                                             parameter.at("ewald_eps"),
                                             static_cast<int>(parameter.at("ewald_arate")));
            break;
        case Auxiliary_Func::Fq_type::Type_1:
            chi = Auxiliary_Func::cal_type_1(new_kvec,
                                             static_cast<int>(parameter.at("ewald_qdiv")),
                                             wfc_basis,
                                             parameter.at("ewald_lambda"));
            break;
        default:
            throw(ModuleBase::GlobalFunc::TO_STRING(__FILE__) + " line " + ModuleBase::GlobalFunc::TO_STRING(__LINE__));
        }
        cal_kernal = std::bind(&Auxiliary_Func::cal_hf_kernel, std::placeholders::_1, chi);
        break;
    case Auxiliary_Func::Kernal_Type::Erfc:
        cal_kernal = std::bind(&Auxiliary_Func::cal_erfc_kernel, std::placeholders::_1, parameter.at("hse_omega"));
        break;
    default:
        throw(ModuleBase::GlobalFunc::TO_STRING(__FILE__) + " line " + ModuleBase::GlobalFunc::TO_STRING(__LINE__));
        break;
    }

    std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>,
              std::vector<std::vector<ModuleBase::ComplexMatrix>>>
        result1 = get_orb_q(new_kvec, wfc_basis, abfs, parameter.at("ewald_ecut"));
    std::vector<std::vector<ModuleBase::Vector3<double>>> gks = result1.first;
    std::vector<std::vector<ModuleBase::ComplexMatrix>> abfs_in_Gs = result1.second;

    std::set<TA> unique_set_A1;
    for (const auto& pair: list_A1)
        unique_set_A1.insert(pair.first);
    std::vector<TA> unique_list_A1(unique_set_A1.begin(), unique_set_A1.end());

    std::map<int, int> abfs_nw = Exx_Abfs::Construct_Orbs::get_nw(abfs);
    std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>> datas;
    datas.resize(nks0);
    for (size_t ik = 0; ik != nks0; ++ik)
    {
        const int npw = gks[ik].size();
        std::vector<ModuleBase::Vector3<double>> gk = gks[ik];
        std::vector<double> vg = cal_kernal(gk);

#pragma omp parallel
        for (size_t i0 = 0; i0 != list_A0.size(); ++i0)
        {
#pragma omp for schedule(dynamic) nowait
            for (size_t i1 = 0; i1 != unique_list_A1.size(); ++i1)
            {
                const TA iat0 = list_A0[i0];
                const int it0 = ucell.iat2it[iat0];
                const int ia0 = ucell.iat2ia[iat0];
                const ModuleBase::Vector3<double> tau0 = ucell.atoms[it0].tau[ia0];

                const TA iat1 = unique_list_A1[i1];
                const int ia1 = ucell.iat2ia[iat1];
                const int it1 = ucell.iat2it[iat1];
                const ModuleBase::Vector3<double> tau1 = ucell.atoms[it1].tau[ia1];

                const size_t abfs_nw_t0 = abfs_nw[it0];
                const size_t abfs_nw_t1 = abfs_nw[it1];
                RI::Tensor<std::complex<double>> data({abfs_nw_t0, abfs_nw_t1});

                for (size_t iw0 = 0; iw0 != abfs_nw_t0; ++iw0)
                    for (size_t iw1 = 0; iw1 != abfs_nw_t1; ++iw1)
                    {
                        for (size_t ig = 0; ig != npw; ++ig)
                        {
                            std::complex<double> phase
                                = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (gk[ig] * (tau0 - tau1)));
                            data(iw0, iw1) += std::conj(abfs_in_Gs[ik][it0](iw0, ig)) * abfs_in_Gs[ik][it1](iw1, ig)
                                              * phase * vg[ig];
                            
                        }
                    }
#pragma omp critical(Ewald_Vq_cal_Vq_q)
                datas[ik][iat0][iat1] = data;
            }
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_q");
    return datas;
}

template <typename Tdata>
std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>, std::vector<std::vector<ModuleBase::ComplexMatrix>>>
    Ewald_Vq<Tdata>::get_orb_q(std::vector<ModuleBase::Vector3<double>>& kvec_c,
                               const ModulePW::PW_Basis_K* wfc_basis,
                               const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
                               const double& gk_ecut)
{
    ModuleBase::TITLE("Ewald_Vq", "get_orb_q");
    ModuleBase::timer::tick("Ewald_Vq", "get_orb_q");

    const int nks0 = kvec_c.size();
    int nmax_total = Exx_Abfs::Construct_Orbs::get_nmax_total(orb_in);
    const int ntype = orb_in.size();
    ModuleBase::realArray table_local(ntype, nmax_total, GlobalV::NQX);
    Wavefunc_in_pw::make_table_q(orb_in, table_local);

    std::vector<std::vector<ModuleBase::ComplexMatrix>> orb_in_Gs(nks0);
    std::vector<int> npwk = get_npwk(kvec_c, wfc_basis, gk_ecut);
    std::vector<std::vector<ModuleBase::Vector3<double>>> gks(nks0);
    std::vector<std::vector<ModuleBase::Vector3<double>>> gcar = get_gcar(npwk, wfc_basis);

    for (size_t ik = 0; ik != nks0; ++ik)
    {
        const int npw = npwk[ik];
        gks[ik].resize(npw);
        for (size_t ig = 0; ig != npw; ++ig)
            gks[ik][ig] = kvec_c[ik] - gcar[ik][ig];

        orb_in_Gs[ik] = produce_local_basis_in_pw(ik, gks[ik], wfc_basis->tpiba, orb_in, table_local);
    }

    std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>,
              std::vector<std::vector<ModuleBase::ComplexMatrix>>>
        result = std::make_pair(gks, orb_in_Gs);

    ModuleBase::timer::tick("Ewald_Vq", "get_orb_q");
    return result;
}

template <typename Tdata>
std::vector<ModuleBase::ComplexMatrix> Ewald_Vq<Tdata>::produce_local_basis_in_pw(
    const int& ik,
    const std::vector<ModuleBase::Vector3<double>>& gk,
    const double& tpiba,
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
    const ModuleBase::realArray& table_local)
{
    ModuleBase::TITLE("Ewald_Vq", "produce_local_basis_in_pw");
    ModuleBase::timer::tick("Ewald_Vq", "produce_local_basis_in_pw");

    const int npw = gk.size();
    const int ntype = orb_in.size();
    std::map<int, int> orb_nw = Exx_Abfs::Construct_Orbs::get_nw(orb_in);
    std::vector<ModuleBase::ComplexMatrix> psi;
    psi.resize(ntype);

    int lmax = std::numeric_limits<int>::min();
    for (const auto& out_vec: orb_in)
        for (const auto& value: out_vec)
        {
            int temp = value.size();
            if (temp > lmax)
                lmax = temp;
        }

    const int total_lm = (lmax + 1) * (lmax + 1);
    ModuleBase::matrix ylm(total_lm, npw);
    ModuleBase::YlmReal::Ylm_Real(total_lm, npw, gk.data(), ylm);

    std::vector<double> flq(npw);
    for (size_t T = 0; T != ntype; ++T)
    {
        ModuleBase::ComplexMatrix sub_psi(orb_nw[T], npw);
        int iwall = 0;
        int ic = 0;
        for (size_t L = 0; L != orb_in[T].size(); ++L)
        {
            std::complex<double> lphase = pow(ModuleBase::NEG_IMAG_UNIT, L);
            for (size_t N = 0; N != orb_in[T][L].size(); ++N)
            {
                for (size_t ig = 0; ig != npw; ++ig)
                    flq[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(table_local,
                                                                            T,
                                                                            ic,
                                                                            GlobalV::NQX,
                                                                            GlobalV::DQ,
                                                                            gk[ig].norm() * tpiba);

                for (size_t m = 0; m != 2 * L + 1; ++m)
                {
                    const int lm = L * L + m;
                    for (size_t ig = 0; ig != npw; ++ig)
                        sub_psi(iwall, ig) = lphase * ylm(lm, ig) * flq[ig];

                    ++iwall;
                }
                ++ic;
            } // end for N
        }     // end for L
        psi[T] = sub_psi;
    } // end for T

    ModuleBase::timer::tick("Ewald_Vq", "produce_local_basis_in_pw");
    return psi;
}

template <typename Tdata>
std::vector<int> Ewald_Vq<Tdata>::get_npwk(std::vector<ModuleBase::Vector3<double>>& kvec_c,
                                           const ModulePW::PW_Basis_K* wfc_basis,
                                           const double& gk_ecut)
{
    const int nks = kvec_c.size();
    std::vector<int> full_npw(nks, wfc_basis->npw);
    std::vector<std::vector<ModuleBase::Vector3<double>>> gcar = get_gcar(full_npw, wfc_basis);
    std::vector<int> npwk(nks);

    for (size_t ik = 0; ik != nks; ++ik)
    {
        int ng = 0;
        for (size_t ig = 0; ig != wfc_basis->npw; ++ig)
        {
            const double gk2 = (gcar[ik][ig] + kvec_c[ik]).norm2();
            if (gk2 <= gk_ecut / wfc_basis->tpiba2)
                ++ng;
        }
        npwk[ik] = ng;
    }

    return npwk;
}

template <typename Tdata>
std::vector<std::vector<int>> Ewald_Vq<Tdata>::get_igl2isz_k(const std::vector<int>& npwk,
                                                             const ModulePW::PW_Basis_K* wfc_basis)
{
    const int nks0 = npwk.size();
    std::vector<std::vector<int>> igl2isz_k(nks0);
    for (size_t ik = 0; ik != nks0; ++ik)
    {
        const int npw = npwk[ik];
        igl2isz_k[ik].resize(npw);
        for (size_t ig = 0; ig != npw; ++ig)
            igl2isz_k[ik][ig] = wfc_basis->ig2isz[ig];
    }

    return igl2isz_k;
}

template <typename Tdata>
std::vector<std::vector<ModuleBase::Vector3<double>>> Ewald_Vq<Tdata>::get_gcar(const std::vector<int>& npwk,
                                                                                const ModulePW::PW_Basis_K* wfc_basis)
{
    const int nks0 = npwk.size();
    std::vector<std::vector<int>> igl2isz_k = get_igl2isz_k(npwk, wfc_basis);
    std::vector<std::vector<ModuleBase::Vector3<double>>> gcar(nks0);
    for (size_t ik = 0; ik != nks0; ++ik)
    {
        const int npw = npwk[ik];
        gcar[ik].resize(npw);
        for (size_t ig = 0; ig != npw; ++ig)
        {
            int isz = igl2isz_k[ik][ig];
            int iz = isz % wfc_basis->nz;
            int is = isz / wfc_basis->nz;
            int ix = wfc_basis->is2fftixy[is] / wfc_basis->fftny;
            int iy = wfc_basis->is2fftixy[is] % wfc_basis->fftny;
            if (ix >= int(wfc_basis->nx / 2) + 1)
                ix -= wfc_basis->nx;
            if (iy >= int(wfc_basis->ny / 2) + 1)
                iy -= wfc_basis->ny;
            if (iz >= int(wfc_basis->nz / 2) + 1)
                iz -= wfc_basis->nz;
            ModuleBase::Vector3<double> f;
            f.x = ix;
            f.y = iy;
            f.z = iz;
            gcar[ik][ig] = f * wfc_basis->G;
        }
    }

    return gcar;
}

template <typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq_R(const K_Vectors* kv,
                               const UnitCell& ucell,
                               std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs)
    -> std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>>
{
    ModuleBase::TITLE("Ewald_Vq", "cal_Vq_R");
    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_R");

    const int nks0 = kv->nks / this->nspin0;
    std::vector<std::map<TA, std::map<TA, RI::Tensor<std::complex<double>>>>> datas;
    datas.resize(nks0);

    for (size_t ik = 0; ik != nks0; ++ik)
    {
        for (const auto& Vs_tmpA: Vs)
        {
            const TA& iat0 = Vs_tmpA.first;
            for (const auto& Vs_tmpB: Vs_tmpA.second)
            {
                const TA& iat1 = Vs_tmpB.first.first;
                const TC& cell1 = Vs_tmpB.first.second;
                std::complex<double> phase
                    = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT
                               * (kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1) * ucell.latvec)));
                if (datas[ik][iat0][iat1].empty())
                    datas[ik][iat0][iat1]
                        = RI::Global_Func::convert<std::complex<double>>(Vs[iat0][Vs_tmpB.first]) * phase;
                else
                    datas[ik][iat0][iat1]
                        = datas[ik][iat0][iat1]
                          + RI::Global_Func::convert<std::complex<double>>(Vs[iat0][Vs_tmpB.first]) * phase;
            }
        }
    }

    ModuleBase::timer::tick("Ewald_Vq", "cal_Vq_R");
    return datas;
}

#endif
