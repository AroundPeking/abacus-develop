//=======================
// AUTHOR : jiyy
// DATE :   2024-01-01
//=======================

#ifndef EWALD_VQ_CPP
#define EWALD_VQ_CPP

#include "ewald_Vq.h"

#include <cmath>

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
double Ewald_Vq::solve_chi(const std::vector<ModuleBase::Vector3<double>>& gk, const T_cal_fq& func_cal_fq)
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
    {
        if (gk[ig].norm2())
            fq_sum += func_cal_fq(gk[ig]) * kv->wk[ik] * SPIN_multiple;
    }

    // cal fq integral
    double fq_int = 0;

    double chi = ModuleBase::FOUR_PI * (fq_int - fq_sum);

    return chi;
}

double Ewald_Vq::fq_type_2(const ModuleBase::Vector3<double>& qvec)
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
    const double prefactor = ModuleBase::TWO_PI * ModuleBase::TWO_PI;

    for(size_t i=0; i!=3; ++i)
    {
        baq[i] = GlobalC::ucell.tpiba * bvec[i] * std::sin(avec[i] * qvec[i] * ModuleBase::TWO_PI);
        baq_2[i] = GlobalC::ucell.tpiba * bvec[i] * std::sin(avec[i] * qvec[i] * ModuleBase::PI);
    }    

    double sum_baq = 0;
    double sum_baq_2 = 0;
    for(size_t i=1; i!=4; ++i)
    {
        size_t j = i%3+1;
        size_t new_i = i-1;
        size_t new_j = j-1;
        sum_baq_2 += baq_2[new_i] * baq_2[new_i];
        sum_baq += baq[new_i] * baq[new_j];
    }
    double fq = prefactor / (4 * sum_baq_2 + 2 * sum_baq);

    return fq;
}

#endif