//=======================
// AUTHOR : jiyy
// DATE :   2024-01-21
//=======================

#ifndef EWALD_VQ_TOOLS_CPP
#define EWALD_VQ_TOOLS_CPP

#include "ewald_Vq_tools.h"

#include "exx_abfs-construct_orbs.h"
#include "module_base/math_polyint.h"
#include "module_base/math_ylmreal.h"

std::vector<ModuleBase::ComplexMatrix> Ewald_Vq_tools::produce_local_basis_in_pw(
    const int& ik,
    const std::vector<ModuleBase::Vector3<double>>& gk,
    const double& tpiba,
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
    const ModuleBase::realArray& table_local)
{
    ModuleBase::TITLE("Ewald_Vq_tools", "produce_local_basis_in_pw");
    ModuleBase::timer::tick("Ewald_Vq_tools", "produce_local_basis_in_pw");

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

    ModuleBase::timer::tick("Ewald_Vq_tools", "produce_local_basis_in_pw");
    return psi;
}

std::vector<int> Ewald_Vq_tools::get_npwk(std::vector<ModuleBase::Vector3<double>>& kvec_c,
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

std::vector<std::vector<int>> Ewald_Vq_tools::get_igl2isz_k(const std::vector<int>& npwk,
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

std::vector<std::vector<ModuleBase::Vector3<double>>> Ewald_Vq_tools::get_gcar(const std::vector<int>& npwk,
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

#endif