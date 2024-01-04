//=======================
// AUTHOR : jiyy
// DATE :   2024-01-01
//=======================

#ifndef EWALD_VQ_CPP
#define EWALD_VQ_CPP

#include "ewald_Vq.h"

#include <cmath>
#include <numeric>

std::vector<double> Ewald_Vq::cal_erfc_kernel(const std::vector<ModuleBase::Vector3<double>>& gk, const double& omega)
{
    const int npw = gk.size();
    std::vector<double> vg(npw);
    for (size_t ig = 0; ig != npw; ++ig)
    {
        if (gk[ig].norm2())
            vg[ig] = (ModuleBase::FOUR_PI / (gk[ig].norm2() * GlobalC::ucell.tpiba2))
                     * (1 - std::exp(-gk[ig].norm2() * GlobalC::ucell.tpiba2 / (4 * omega * omega)));
        else
            vg[ig] = ModuleBase::FOUR_PI / (4 * omega * omega);
    }

    return vg;
}

std::vector<double> Ewald_Vq::cal_hf_kernel(const std::vector<ModuleBase::Vector3<double>>& gk)
{
    const int npw = gk.size();
    std::vector<double> vg(npw);
    for (size_t ig = 0; ig != npw; ++ig)
    {
        if (gk[ig].norm2())
            vg[ig] = ModuleBase::FOUR_PI / (gk[ig].norm2() * GlobalC::ucell.tpiba2);
        else // set 0 for add Auxiliary functions to eliminate singularities later
            vg[ig] = 0;
    }

    return vg;
}

// for numerical integral of fq
double Ewald_Vq::solve_chi(const std::vector<ModuleBase::Vector3<double>>& gk,
                           const T_cal_fq<double>& func_cal_fq,
                           const TC nq_vec,
                           const int& niter,
                           const double& eps,
                           const int& a_rate)
{
    const double SPIN_multiple = std::map<int, double>{
        {1, 0.5},
        {2, 1  },
        {4, 1  }
    }.at(GlobalV::NSPIN);
    const int npw = gk.size();

    // cal fq sum except q=0
    double fq_sum = 0;
    for (size_t ig = 0; ig != npw; ++ig)
        if (gk[ig].norm2())
            fq_sum += func_cal_fq(gk[ig]) * kv->wk[ik] * SPIN_multiple;

    // cal fq integral
    double fq_int = Iter_Integral(func_cal_fq, nq_vec, niter, eps, a_rate);

    double chi = ModuleBase::FOUR_PI * (fq_int - fq_sum);

    return chi;
}

double Ewald_Vq::fq_type_1(const ModuleBase::Vector3<double>& qvec, const int& qdiv)
{
    std::vector<ModuleBase::Vector3<double>> avec = {GlobalC::ucell.a1, GlobalC::ucell.a2, GlobalC::ucell.a3};
    std::vector<ModuleBase::Vector3<double>> bvec;
    bvec.resize(3);
    bvec[0].x = GlobalC::ucell.G.e11;
    bvec[0].y = GlobalC::ucell.G.e12;
    bvec[0].z = GlobalC::ucell.G.e13;

    bvec[1].x = GlobalC::ucell.G.e21;
    bvec[1].y = GlobalC::ucell.G.e22;
    bvec[1].z = GlobalC::ucell.G.e23;

    bvec[2].x = GlobalC::ucell.G.e31;
    bvec[2].y = GlobalC::ucell.G.e32;
    bvec[2].z = GlobalC::ucell.G.e33;

    std::vector<double> baq(3);
    std::vector<double> baq_2(3);
    const int qexpo = -abs(qdiv);
    const double prefactor = std::pow(ModuleBase::TWO_PI, -qexpo);

    for (size_t i = 0; i != 3; ++i)
    {
        baq[i] = GlobalC::ucell.tpiba * bvec[i] * std::sin(avec[i] * qvec[i] * ModuleBase::TWO_PI);
        baq_2[i] = GlobalC::ucell.tpiba * bvec[i] * std::sin(avec[i] * qvec[i] * ModuleBase::PI);
    }

    double sum_baq = 0;
    double sum_baq_2 = 0;
    for (size_t i = 1; i != 4; ++i)
    {
        size_t j = i % 3 + 1;
        size_t new_i = i - 1;
        size_t new_j = j - 1;
        sum_baq_2 += baq_2[new_i] * baq_2[new_i];
        sum_baq += baq[new_i] * baq[new_j];
    }
    double fq = prefactor * std::pow(4 * sum_baq_2 + 2 * sum_baq, 0.5 * qexpo);

    return fq;
}

double Ewald_Vq::cal_type_1(const std::vector<ModuleBase::Vector3<double>>& gk, const int& qdiv)
{
    const T_cal_fq<double> func_cal_fq_type_1 = std::bind(&fq_type_1, this, std::placeholders::_1, qdiv);
    return this->solve_chi(gk, func_cal_fq_type_1);
}

double Ewald_Vq::Iter_Integral(const T_cal_fq<double>& func_cal_fq,
                               const TC nq_vec,
                               const int& niter,
                               const double& eps,
                               const int& a_rate)
{
    bool any_negative = std::any_of(nq_vec.begin(), nq_vec.end(), [](int i) { return i < 0; });
    bool any_nthree = std::any_of(nq_vec.begin(), nq_vec.end(), [](int i) { return i % 3 != 0; });
    if (any_negative || any_nthree)
        WARNING_QUIT("Ewald_Vq::Iter_Integral",
                     "The elements of `nq_vec` should be non-negative and multiples of three!");
    bool all_zero = std::all_of(nq_vec.begin(), nq_vec.end(), [](int i) { return i == 0; });
    if (all_zero)
        WARNING_QUIT("Ewald_Vq::Iter_Integral", "At least one element of `nq_vec` should be non-zero!");

    const int nqs
        = std::accumulate(nq_vec.begin(), nq_vec.end(), 1, [](int a, int b) { return (2 * a + 1) * (2 * b + 1); });
    std::array<double, 3> qstep{};
    TC nq_vec_in{};
    int ndim = 0;
    for (size_t i = 0; i != 3; ++i)
    {
        if (nq_vec[i] != 0)
        {
            qstep[i] = 1 / (2 * nq_vec[i] + 1);
            ndim += 1;
        }
        nq_vec_in[i] = nq_vec[i] / a_rate;
    }

    double integ = 0.0;
    for (size_t iter = 0; iter != niter; ++iter)
    {
        double integ_iter = 0.0;
        for (size_t ig1 = -nq_vec[0]; ig1 == nq_vec[0]; ++ig1)
        {
            for (size_t ig2 = -nq_vec[1]; ig2 == nq_vec[1]; ++ig2)
            {
                for (size_t ig3 = -nq_vec[2]; ig3 == nq_vec[2]; ++ig3)
                {
                    if (std::abs(ig1) <= nq_vec_in[0] && std::abs(ig2) <= nq_vec_in[1] && std::abs(ig3) <= nq_vec_in[2])
                        continue;
                    ModuleBase::Vector3<double> qvec;
                    qvec.x = qstep[0] * static_cast<double>(ig1);
                    qvec.y = qstep[1] * static_cast<double>(ig2);
                    qvec.z = qstep[2] * static_cast<double>(ig3);
                    integ_iter += func_cal_fq(qvec);
                }
            }
        }
        integ_iter /= nqs * pow(a_rate, ndim * (iter - 1)); // Each iteration reduces dq by a multiple of a_rate
        integ += integ_iter;
        if (iter != 0 && integ_iter < eps)
            break;
        for (auto& qs: qstep)
            qs /= a_rate;
    }

    if (iter == niter)
        WARNING_QUIT("Ewald_Vq::Iter_Integral", "Integral not converged!");

    return integ;
}

#endif