// This file contains subroutines realted to gradient calculations
// it contains 5 subroutines:
// 1. gradcorr, which calculates gradient correction
// 2. grad_wfc, which calculates gradient of wavefunction
//		it is used in stress_func_mgga.cpp
// 3. grad_rho, which calculates gradient of density
// 4. grad_dot, which calculates divergence of something
// 5. noncolin_rho, which diagonalizes the spin density matrix
//  and gives the spin up and spin down components of the charge.

#include "module_base/timer.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "xc_functional.h"

#include <ATen/core/tensor.h>
#include <ATen/core/tensor_map.h>
#include <ATen/core/tensor_types.h>
#include <module_hamilt_general/module_xc/kernels/xc_functional_op.h>
#include "module_parameter/parameter.h"

int XC_Functional::gga_grad = 1;

// from gradcorr.f90
void XC_Functional::gradcorr(double& etxc,
                             double& vtxc,
                             ModuleBase::matrix& v,
                             const Charge* const chr,
                             ModulePW::PW_Basis* rhopw,
                             const UnitCell* ucell,
                             std::vector<double>& stress_gga,
                             const bool is_stress)
{
    ModuleBase::TITLE("XC_Functional", "gradcorr");

    if (func_type == 0 || func_type == 1)
    {
        return; // none or LDA functional
    }

    bool igcc_is_lyp = false;
    if (func_id[1] == XC_GGA_C_LYP)
    {
        igcc_is_lyp = true;
    }

    int nspin0 = GlobalV::NSPIN;
    if (GlobalV::NSPIN == 4)
    {
        nspin0 = 1;
    }
    if (GlobalV::NSPIN == 4 && (GlobalV::DOMAG || GlobalV::DOMAG_Z))
    {
        nspin0 = 2;
    }

    assert(nspin0 > 0);
    const double fac0 = GlobalV::NSPIN == 2 ? 0.5 : 1.0;
    const double fac = 1.0 / nspin0;

    if (is_stress)
    {
        stress_gga.resize(9);
        for (int i = 0; i < 9; i++)
        {
            stress_gga[i] = 0.0;
        }
    }

    // doing FFT to get rho in G space: rhog1
    rhopw->real2recip(chr->rho[0], chr->rhog[0]);
    if (GlobalV::NSPIN == 2)
    {
        rhopw->real2recip(chr->rho[1], chr->rhog[1]);
    }
    rhopw->real2recip(chr->rho_core, chr->rhog_core);

    // sum up (rho_core+rho) for each spin in real space
    // and reciprocal space.
    double* rhotmp1 = nullptr;                   // rho_up or rho
    double* rhotmp2 = nullptr;                   // rho_down
    std::complex<double>* rhogsum1 = nullptr;    // rhog_up or rhog
    std::complex<double>* rhogsum2 = nullptr;    // rhog_down
    ModuleBase::Vector3<double>* gdr1 = nullptr; // gradient of rho_up or rho
    ModuleBase::Vector3<double>* gdr2 = nullptr; // gradient of rho_down
    ModuleBase::Vector3<double>* h1 = nullptr;   // for gradient correction
    ModuleBase::Vector3<double>* h2 = nullptr;   // for gradient correction
    double** vsave = nullptr;
    double** vgg = nullptr;

    // for spin unpolarized case,
    // calculate the gradient of (rho_core+rho) in reciprocal space.
    rhotmp1 = new double[rhopw->nrxx];
    rhogsum1 = new std::complex<double>[rhopw->npw];
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int ir = 0; ir < rhopw->nrxx; ir++)
    {
        rhotmp1[ir] = chr->rho[0][ir] + fac0 * chr->rho_core[ir];
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int ig = 0; ig < rhopw->npw; ig++)
    {
        rhogsum1[ig] = chr->rhog[0][ig] + fac0 * chr->rhog_core[ig];
    }

    gdr1 = new ModuleBase::Vector3<double>[rhopw->nrxx];
    if (!is_stress)
    {
        h1 = new ModuleBase::Vector3<double>[rhopw->nrxx];
    }

    XC_Functional::grad_rho(rhogsum1, gdr1, rhopw, ucell->tpiba);

    // for spin polarized case;
    // calculate the gradient of (rho_core+rho) in reciprocal space.
    if (GlobalV::NSPIN == 2)
    {
        rhotmp2 = new double[rhopw->nrxx];
        rhogsum2 = new std::complex<double>[rhopw->npw];
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            rhotmp2[ir] = chr->rho[1][ir] + fac * chr->rho_core[ir];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < rhopw->npw; ig++)
        {
            rhogsum2[ig] = chr->rhog[1][ig] + fac * chr->rhog_core[ig];
        }

        gdr2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
        if (!is_stress)
        {
            h2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
        }

        XC_Functional::grad_rho(rhogsum2, gdr2, rhopw, ucell->tpiba);
    }

    // new method to calculate gradient of magnetic density
    std::vector<double> mag_part; // to save mag.x/|mag|
    std::complex<double>* tmp_recip = nullptr;      // to save mag or rho in reciprocal space for each spin
    ModuleBase::Vector3<double>* gdr_mag = nullptr; // to save gradient of mag or rho for each spin
    if (GlobalV::NSPIN == 4 && (GlobalV::DOMAG || GlobalV::DOMAG_Z))
    {
        mag_part.resize(rhopw->nrxx * 3, 0.0);
        rhotmp2 = new double[rhopw->nrxx];
        rhogsum2 = new std::complex<double>[rhopw->npw];
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            rhotmp1[ir] = 0.0;
            rhotmp2[ir] = 0.0;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < rhopw->npw; ig++)
        {
            rhogsum1[ig] = 0.0;
            rhogsum2[ig] = 0.0;
        }
        if (!is_stress)
        {
            vsave = new double*[GlobalV::NSPIN];
            for (int is = 0; is < GlobalV::NSPIN; is++)
            {
                vsave[is] = new double[rhopw->nrxx];
            }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
            for (int is = 0; is < GlobalV::NSPIN; is++)
            {
                for (int ir = 0; ir < rhopw->nrxx; ir++)
                {
                    vsave[is][ir] = v(is, ir);
                    v(is, ir) = 0;
                }
            }
            vgg = new double*[nspin0];
            for (int is = 0; is < nspin0; is++)
            {
                vgg[is] = new double[rhopw->nrxx];
            }
        }
        noncolin_rho(rhotmp1, rhotmp2, chr->rho, rhopw->nrxx, mag_part.data());
        rhopw->real2recip(rhotmp1, rhogsum1);
        rhopw->real2recip(rhotmp2, rhogsum2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            rhotmp2[ir] += fac * chr->rho_core[ir];
            rhotmp1[ir] += fac * chr->rho_core[ir];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < rhopw->npw; ig++)
        {
            rhogsum2[ig] += fac * chr->rhog_core[ig];
            rhogsum1[ig] += fac * chr->rhog_core[ig];
        }

        gdr2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
        if (!is_stress)
            h2 = new ModuleBase::Vector3<double>[rhopw->nrxx];

        if (XC_Functional::gga_grad == 2)
        {
        // for non-collinear case
        // calculate the gradient of
        // 1. rho' = rho_core+rho
        // 2. mag_x, mag_y, mag_z
        // rho_up = 0.5 rho' + 0.5 |mag|
        // rho_down = 0.5 rho' - 0.5 |mag|
        // \nabla rho_up = 0.5 \nabla rho' + 0.5 (\sum_i \nabla mag_i mag_i/|mag|)
        // \nabla rho_down = 0.5 \nabla rho' - 0.5 (\sum_i \nabla mag_i mag_i/|mag|)
        // rho' has been calculated in rhotmp1, rhog' has been calculated in rhogsum1, \nabla rho' has been calculated
        // in gdr1
        tmp_recip = new std::complex<double>[rhopw->npw];
        gdr_mag = new ModuleBase::Vector3<double>[rhopw->nrxx];
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            gdr_mag[ir] = gdr1[ir];
            gdr1[ir] = 0.5 * gdr_mag[ir];
            gdr2[ir] = 0.5 * gdr_mag[ir];
        }
        for (int is = 1; is < 4; is++)
        {
            rhopw->real2recip(chr->rho[is], tmp_recip);
            XC_Functional::grad_rho(tmp_recip, gdr_mag, rhopw, ucell->tpiba);
            const double* mag_part_is = mag_part.data() + (is-1) * rhopw->nrxx;
            for (int ir = 0; ir < rhopw->nrxx; ir++)
            {
                const ModuleBase::Vector3<double> grad_is = 0.5 * gdr_mag[ir] * mag_part_is[ir];
                gdr1[ir] += grad_is;
                gdr2[ir] -= grad_is;
            }
        }
        delete[] tmp_recip;
        delete[] gdr_mag;
        }
        else
        {
            XC_Functional::grad_rho(rhogsum1, gdr1, rhopw, ucell->tpiba);
            XC_Functional::grad_rho(rhogsum2, gdr2, rhopw, ucell->tpiba);
        }
    }

    const double epsr = 1.0e-6;
    const double epsg = 1.0e-10;

    double vtxcgc = 0.0;
    double etxcgc = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<double> local_stress_gga;
        double local_vtxcgc = 0.0;
        double local_etxcgc = 0.0;

        if (is_stress)
        {
            local_stress_gga.resize(9);
            for (int i = 0; i < 9; i++)
            {
                local_stress_gga[i] = 0.0;
            }
        }
#else
    std::vector<double>& local_stress_gga = stress_gga;
    double& local_vtxcgc = vtxcgc;
    double& local_etxcgc = etxcgc;
#endif

        double grho2a = 0.0;
        double grho2b = 0.0;
        double sxc = 0.0;
        double v1xc = 0.0;
        double v2xc = 0.0;

        if (nspin0 == 1)
        {
            double segno;
#ifdef _OPENMP
#pragma omp for
#endif
            for (int ir = 0; ir < rhopw->nrxx; ir++)
            {
                const double arho = std::abs(rhotmp1[ir]);
                if (!is_stress)
                {
                    h1[ir].x = h1[ir].y = h1[ir].z = 0.0;
                }

                if (arho > epsr)
                {
                    grho2a = gdr1[ir].norm2();

                    // normally values in rhotmp can either be >= 0 or < 0.
                    if (rhotmp1[ir] >= 0.0)
                    {
                        segno = 1.0;
                    }
                    else
                    {
                        segno = -1.0;
                    }
                    if (use_libxc && is_stress)
                    {
#ifdef USE_LIBXC
                        if (func_type == 3 || func_type == 5) // the gradcorr part to stress of mGGA
                        {
                            double v3xc;
                            double atau = chr->kin_r[0][ir] / 2.0;
                            XC_Functional::tau_xc(arho, grho2a, atau, sxc, v1xc, v2xc, v3xc);
                        }
                        else
                        {
                            XC_Functional::gcxc_libxc(arho, grho2a, sxc, v1xc, v2xc);
                        }
#endif
                    } // end use_libxc
                    else
                    {
                        XC_Functional::gcxc(arho, grho2a, sxc, v1xc, v2xc);
                    }
                    if (is_stress)
                    {
                        double tt[3];
                        tt[0] = gdr1[ir].x;
                        tt[1] = gdr1[ir].y;
                        tt[2] = gdr1[ir].z;
                        for (int l = 0; l < 3; l++)
                        {
                            for (int m = 0; m < l + 1; m++)
                            {
                                int ind = l * 3 + m;
                                local_stress_gga[ind] += tt[l] * tt[m] * ModuleBase::e2 * v2xc;
                            }
                        }
                    }
                    else
                    {
                        // first term of the gradient correction:
                        // D(rho*Exc)/D(rho)
                        v(0, ir) += ModuleBase::e2 * v1xc;
                        // cout << "v    " << v(0, ir) << endl;

                        // h contains
                        // D(rho*Exc) / D(|grad rho|) * (grad rho) / |grad rho|
                        h1[ir] = ModuleBase::e2 * v2xc * gdr1[ir];

                        local_vtxcgc += ModuleBase::e2 * v1xc * (rhotmp1[ir] - chr->rho_core[ir]);
                        local_etxcgc += ModuleBase::e2 * sxc * segno;
                    }
                } // end arho > epsr
            }
        } // end nspin0 == 1
        else // spin polarized case
        {
#ifdef _OPENMP
#pragma omp for
#endif
            for (int ir = 0; ir < rhopw->nrxx; ir++)
            {
                if (use_libxc)
                {
#ifdef USE_LIBXC
                    double sxc, v1xcup, v1xcdw, v2xcup, v2xcdw, v2xcud;
                    if (func_type == 3 || func_type == 5) // the gradcorr part to stress of mGGA
                    {
                        double v3xcup, v3xcdw;
                        double atau1 = chr->kin_r[0][ir] / 2.0;
                        double atau2 = chr->kin_r[1][ir] / 2.0;
                        XC_Functional::tau_xc_spin(rhotmp1[ir],
                                                   rhotmp2[ir],
                                                   gdr1[ir],
                                                   gdr2[ir],
                                                   atau1,
                                                   atau2,
                                                   sxc,
                                                   v1xcup,
                                                   v1xcdw,
                                                   v2xcup,
                                                   v2xcdw,
                                                   v2xcud,
                                                   v3xcup,
                                                   v3xcdw);
                    }
                    else
                    {
                        XC_Functional::gcxc_spin_libxc(rhotmp1[ir],
                                                       rhotmp2[ir],
                                                       gdr1[ir],
                                                       gdr2[ir],
                                                       sxc,
                                                       v1xcup,
                                                       v1xcdw,
                                                       v2xcup,
                                                       v2xcdw,
                                                       v2xcud);
                    }
                    if (is_stress)
                    {
                        double tt1[3], tt2[3];
                        {
                            tt1[0] = gdr1[ir].x;
                            tt1[1] = gdr1[ir].y;
                            tt1[2] = gdr1[ir].z;
                            tt2[0] = gdr2[ir].x;
                            tt2[1] = gdr2[ir].y;
                            tt2[2] = gdr2[ir].z;
                        }
                        for (int l = 0; l < 3; l++)
                        {
                            for (int m = 0; m < l + 1; m++)
                            {
                                int ind = l * 3 + m;
                                local_stress_gga[ind] += (tt1[l] * tt1[m] * v2xcup + tt2[l] * tt2[m] * v2xcdw
                                                          + (tt1[l] * tt2[m] + tt2[l] * tt1[m]) * v2xcud)
                                                         * ModuleBase::e2;
                            }
                        }
                    }
                    else
                    {
                        // first term of the gradient correction : D(rho*Exc)/D(rho)
                        v(0, ir) += ModuleBase::e2 * v1xcup;
                        v(1, ir) += ModuleBase::e2 * v1xcdw;

                        // h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
                        h1[ir] += ModuleBase::e2 * (v2xcup * gdr1[ir] + v2xcud * gdr2[ir]);
                        h2[ir] += ModuleBase::e2 * (v2xcdw * gdr2[ir] + v2xcud * gdr1[ir]);

                        local_vtxcgc = local_vtxcgc + ModuleBase::e2 * v1xcup * (rhotmp1[ir] - chr->rho_core[ir] * fac);
                        local_vtxcgc = local_vtxcgc + ModuleBase::e2 * v1xcdw * (rhotmp2[ir] - chr->rho_core[ir] * fac);
                        local_etxcgc = local_etxcgc + ModuleBase::e2 * sxc;
                    }
#endif
                }
                else
                {
                    double v1cup = 0.0;
                    double v1cdw = 0.0;
                    double v2cup = 0.0;
                    double v2cdw = 0.0;
                    double v1xup = 0.0;
                    double v1xdw = 0.0;
                    double v2xup = 0.0;
                    double v2xdw = 0.0;
                    double v2cud = 0.0;
                    double v2c = 0.0;
                    double sx = 0.0;
                    double sc = 0.0;
                    double rh = rhotmp1[ir] + rhotmp2[ir];
                    grho2a = gdr1[ir].norm2();
                    grho2b = gdr2[ir].norm2();
                    XC_Functional::gcx_spin(rhotmp1[ir], rhotmp2[ir], grho2a, grho2b, sx, v1xup, v1xdw, v2xup, v2xdw);

                    if (rh > epsr)
                    {
                        if (igcc_is_lyp)
                        {
                            ModuleBase::WARNING_QUIT("XC_Functional", "igcc_is_lyp is not available now.");
                        }
                        else
                        {
                            double zeta = (rhotmp1[ir] - rhotmp2[ir]) / rh;
                            if (GlobalV::NSPIN == 4 && (GlobalV::DOMAG || GlobalV::DOMAG_Z))
                            {
                                zeta = fabs(zeta);
                            }
                            const double grh2 = (gdr1[ir] + gdr2[ir]).norm2();
                            XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
                            v2cup = v2c;
                            v2cdw = v2c;
                            v2cud = v2c;
                        }
                    }
                    else
                    {
                        sc = 0.0;
                        v1cup = 0.0;
                        v1cdw = 0.0;
                        v2c = 0.0;
                        v2cup = 0.0;
                        v2cdw = 0.0;
                        v2cud = 0.0;
                    }

                    if (is_stress)
                    {
                        double tt1[3], tt2[3];
                        {
                            tt1[0] = gdr1[ir].x;
                            tt1[1] = gdr1[ir].y;
                            tt1[2] = gdr1[ir].z;
                            tt2[0] = gdr2[ir].x;
                            tt2[1] = gdr2[ir].y;
                            tt2[2] = gdr2[ir].z;
                        }
                        for (int l = 0; l < 3; l++)
                        {
                            for (int m = 0; m < l + 1; m++)
                            {
                                int ind = l * 3 + m;
                                //    exchange
                                local_stress_gga[ind] += tt1[l] * tt1[m] * ModuleBase::e2 * v2xup
                                                         + tt2[l] * tt2[m] * ModuleBase::e2 * v2xdw;
                                //    correlation
                                local_stress_gga[ind] += (tt1[l] * tt1[m] * v2cup + tt2[l] * tt2[m] * v2cdw
                                                          + (tt1[l] * tt2[m] + tt2[l] * tt1[m]) * v2cud)
                                                         * ModuleBase::e2;
                            }
                        }
                    }
                    else
                    {
                        // first term of the gradient correction : D(rho*Exc)/D(rho)
                        v(0, ir) = v(0, ir) + ModuleBase::e2 * (v1xup + v1cup);
                        v(1, ir) = v(1, ir) + ModuleBase::e2 * (v1xdw + v1cdw);

                        // h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
                        h1[ir] = ModuleBase::e2 * ((v2xup + v2cup) * gdr1[ir] + v2cud * gdr2[ir]);
                        h2[ir] = ModuleBase::e2 * ((v2xdw + v2cdw) * gdr2[ir] + v2cud * gdr1[ir]);

                        local_vtxcgc
                            = local_vtxcgc + ModuleBase::e2 * (v1xup + v1cup) * (rhotmp1[ir] - chr->rho_core[ir] * fac);
                        local_vtxcgc
                            = local_vtxcgc + ModuleBase::e2 * (v1xdw + v1cdw) * (rhotmp2[ir] - chr->rho_core[ir] * fac);
                        local_etxcgc = local_etxcgc + ModuleBase::e2 * (sx + sc);
                    }
                }
            } // end ir
        }
#ifdef _OPENMP
#pragma omp critical(xc_functional_gradcorr_reduce)
        {
            if (is_stress)
            {
                for (int l = 0; l < 3; l++)
                {
                    for (int m = 0; m < l + 1; m++)
                    {
                        int ind = l * 3 + m;
                        stress_gga[ind] += local_stress_gga[ind];
                    }
                }
            }
            else
            {
                vtxcgc += local_vtxcgc;
                etxcgc += local_etxcgc;
            }
        }
    }
#endif

    // std::cout << "\n vtxcgc=" << vtxcgc;
    // std::cout << "\n etxcgc=" << etxcgc << std::endl;

    if(!is_stress && GlobalV::NSPIN==4 && XC_Functional::gga_grad==2)
    {
        // calculate the grad(h1+h2) and grad((h1-h2)*mag_i/mag_part) respectively
        // and add them to v(0, ir) and v(1-3, ir) respectively.
        // transfer v from v^up and v^down to v^rho, v^mag_x, v^mag_y, v^mag_z
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            double vgg[2] = {v(0, ir), v(1, ir)};
            v(0, ir) = 0.5 * (vgg[0] + vgg[1]) + vsave[0][ir];
            const double vgg_diff = 0.5 * (vgg[0] - vgg[1]);
            for (int i = 1; i < 4; i++)
            {
                v(i, ir) = vgg_diff * mag_part[ir + (i - 1) * rhopw->nrxx] + vsave[i][ir];
            }
        }

        // second term of the gradient correction :
        // \sum_alpha (D / D r_alpha) ( D(rho*Exc)/D(grad_alpha rho) )

        // dh is in real sapce.
        double* dh = new double[rhopw->nrxx];
        double* dh1 = new double[rhopw->nrxx];
        ModuleBase::Vector3<double>* tmp_h = new ModuleBase::Vector3<double>[rhopw->nrxx];
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            tmp_h[ir] = 0.5 * (h1[ir] + h2[ir]);
            dh1[ir] = 0.0;
        }
        XC_Functional::grad_dot(tmp_h, dh, rhopw, ucell->tpiba);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            v(0, ir) -= dh[ir];
        }
        double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static, 256)
#endif
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            sum += dh[ir] * chr->rho[0][ir];
        }
        vtxcgc -= sum;

        for(int is=1;is<4;is++)
        {
            const double* mag_part_is = &mag_part[(is - 1) * rhopw->nrxx];
            for (int ir = 0; ir < rhopw->nrxx; ir++)
            {
                tmp_h[ir] = 0.5 * (h1[ir] - h2[ir]) * mag_part_is[ir];
            }
            XC_Functional::grad_dot(tmp_h, dh, rhopw, ucell->tpiba);
            for(int ir=0;ir<rhopw->nrxx;ir++)
            {
                dh1[ir] += dh[ir] * mag_part_is[ir];
            }
        }
        for(int is=1;is<4;is++)
        {
            const double* mag_part_is = &mag_part[(is - 1) * rhopw->nrxx];
            for(int ir=0;ir<rhopw->nrxx;ir++)
            {
                v(is,ir) -= dh1[ir] * mag_part_is[ir];
            }
        }
        sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static, 256)
#endif
                for (int ir = 0; ir < rhopw->nrxx; ir++)
                {
                    sum += dh1[ir] * (rhotmp1[ir] - rhotmp2[ir]);
                }
                vtxcgc -= sum;

        delete[] dh;
        delete[] dh1;
        delete[] tmp_h;

        vtxc += vtxcgc;
        etxc += etxcgc;

    }
    else if (!is_stress)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rhopw->nrxx; ir++)
        {
            rhotmp1[ir] -= fac * chr->rho_core[ir];
        }
        if (nspin0 == 2)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (int ir = 0; ir < rhopw->nrxx; ir++)
            {
                rhotmp2[ir] -= fac * chr->rho_core[ir];
            }
        }

        // second term of the gradient correction :
        // \sum_alpha (D / D r_alpha) ( D(rho*Exc)/D(grad_alpha rho) )

        // dh is in real sapce.
        double* dh = new double[rhopw->nrxx];

        for (int is = 0; is < nspin0; is++)
        {
            if (is == 0)
            {
                XC_Functional::grad_dot(h1, dh, rhopw, ucell->tpiba);
            }
            if (is == 1)
            {
                XC_Functional::grad_dot(h2, dh, rhopw, ucell->tpiba);
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (int ir = 0; ir < rhopw->nrxx; ir++)
            {
                v(is, ir) -= dh[ir];
            }

            double sum = 0.0;
            if (is == 0)
            {
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static, 256)
#endif
                for (int ir = 0; ir < rhopw->nrxx; ir++)
                {
                    sum += dh[ir] * rhotmp1[ir];
                }
            }
            else if (is == 1)
            {
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static, 256)
#endif
                for (int ir = 0; ir < rhopw->nrxx; ir++)
                {
                    sum += dh[ir] * rhotmp2[ir];
                }
            }
            vtxcgc -= sum;
        }

        delete[] dh;

        vtxc += vtxcgc;
        etxc += etxcgc;

        if(GlobalV::NSPIN == 4 && XC_Functional::gga_grad == 1)
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
            for(int is=0;is<GlobalV::NSPIN;is++)
            {
                for(int ir=0;ir<rhopw->nrxx;ir++)
                {
                    if(is<nspin0) 
                    { 
                        vgg[is][ir] = v(is,ir);
                    }
                    v(is,ir) = vsave[is][ir];
                }
            }
            const double* mag_part_p[3] = {mag_part.data(), mag_part.data() + rhopw->nrxx, mag_part.data() + 2 * rhopw->nrxx};
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for(int ir=0;ir<rhopw->nrxx;ir++)
            {
                v(0,ir) += 0.5 * (vgg[0][ir] + vgg[1][ir]);
                for(int i=1;i<4;i++) 
                {
                    v(i,ir) += 0.5 *(vgg[0][ir]-vgg[1][ir]) * mag_part_p[i-1][ir];
                }
            }
        }
    }

    // deacllocate
    delete[] rhotmp1;
    delete[] rhogsum1;
    delete[] gdr1;
    if (!is_stress)
    {
        delete[] h1;
    }

    if (GlobalV::NSPIN == 2)
    {
        delete[] rhotmp2;
        delete[] rhogsum2;
        delete[] gdr2;
        if (!is_stress)
        {
            delete[] h2;
        }
    }
    if (GlobalV::NSPIN == 4 && (GlobalV::DOMAG || GlobalV::DOMAG_Z))
    {
        if (!is_stress)
        {
            for (int i = 0; i < nspin0; i++)
            {
                delete[] vgg[i];
            }
            delete[] vgg;
            for (int i = 0; i < GlobalV::NSPIN; i++)
            {
                delete[] vsave[i];
            }
            delete[] vsave;
            delete[] h2;
        }
        delete[] rhotmp2;
        delete[] rhogsum2;
        delete[] gdr2;
    }

    return;
}

template <typename T, typename Device, typename Real>
void XC_Functional::grad_wfc(const int ik,
                             const Real tpiba,
                             const ModulePW::PW_Basis_K* wfc_basis,
                             const T* rhog,
                             T* grad)
{
    using ct_Device = typename ct::PsiToContainer<Device>::type;
    const int npw_k = wfc_basis->npwk[ik];

    auto porter = std::move(
        ct::Tensor(ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {wfc_basis->nmaxgr}));
    auto gcar = ct::TensorMap(&wfc_basis->gcar[0][0],
                              ct::DataType::DT_DOUBLE,
                              ct::DeviceType::CpuDevice,
                              {wfc_basis->nks * wfc_basis->npwk_max, 3})
                    .to_device<ct_Device>();
    auto kvec_c = ct::TensorMap(&wfc_basis->kvec_c[0][0],
                                ct::DataType::DT_DOUBLE,
                                ct::DeviceType::CpuDevice,
                                {wfc_basis->nks, 3})
                      .to_device<ct_Device>();

    auto xc_functional_grad_wfc_solver = hamilt::xc_functional_grad_wfc_op<T, Device>();

    for (int ipol = 0; ipol < 3; ipol++)
    {
        xc_functional_grad_wfc_solver(ik,
                                      ipol,
                                      npw_k,
                                      wfc_basis->npwk_max,          // Integers
                                      tpiba,                        // Double
                                      gcar.template data<Real>(),   // Array of Real
                                      kvec_c.template data<Real>(), // Array of double
                                      rhog,
                                      porter.data<T>()); // Array of std::complex<double>

        // bring the gdr from G --> R
        Device* ctx = nullptr;
        wfc_basis->recip_to_real(ctx, porter.data<T>(), porter.data<T>(), ik);

        xc_functional_grad_wfc_solver(ipol,
                                      wfc_basis->nrxx, // Integers
                                      porter.data<T>(),
                                      grad); // Array of std::complex<double>
    }
}

void XC_Functional::grad_rho(const std::complex<double>* rhog,
                             ModuleBase::Vector3<double>* gdr,
                             const ModulePW::PW_Basis* rho_basis,
                             const double tpiba)
{
    std::complex<double>* gdrtmp = new std::complex<double>[rho_basis->nmaxgr];

    // the formula is : rho(r)^prime = \int iG * rho(G)e^{iGr} dG
    for (int i = 0; i < 3; ++i)
    {
        // calculate the charge density gradient in reciprocal space.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < rho_basis->npw; ig++)
        {
            gdrtmp[ig] = ModuleBase::IMAG_UNIT * rhog[ig] * rho_basis->gcar[ig][i];
        }

        // bring the gdr from G --> R
        rho_basis->recip2real(gdrtmp, gdrtmp);

        // remember to multily 2pi/a0, which belongs to G vectors.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rho_basis->nrxx; ir++)
        {
            gdr[ir][i] = gdrtmp[ir].real() * tpiba;
        }
    }

    delete[] gdrtmp;
    return;
}

void XC_Functional::grad_dot(const ModuleBase::Vector3<double>* h,
                             double* dh,
                             const ModulePW::PW_Basis* rho_basis,
                             const double tpiba)
{
    std::complex<double>* aux = new std::complex<double>[rho_basis->nmaxgr];
    std::complex<double>* gaux = new std::complex<double>[rho_basis->npw];

    for (int i = 0; i < 3; ++i)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ir = 0; ir < rho_basis->nrxx; ++ir)
        {
            aux[ir] = std::complex<double>(h[ir][i], 0.0);
        }

        // bring to G space.
        rho_basis->real2recip(aux, aux);
        if (i == 0)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (int ig = 0; ig < rho_basis->npw; ++ig)
            {
                gaux[ig] = ModuleBase::IMAG_UNIT * aux[ig] * rho_basis->gcar[ig][i];
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (int ig = 0; ig < rho_basis->npw; ++ig)
            {
                gaux[ig] += ModuleBase::IMAG_UNIT * aux[ig] * rho_basis->gcar[ig][i];
            }
        }
    }

    // bring back to R space
    rho_basis->recip2real(gaux, aux);

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        dh[ir] = aux[ir].real() * tpiba;
    }

    delete[] aux;
    delete[] gaux;
    return;
}

void XC_Functional::noncolin_rho(double* rhoout1,
                                 double* rhoout2,
                                 const double* const* const rho,
                                 const int nrxx,
                                 double* mag_part)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int ir = 0; ir < nrxx; ir++)
    {
        double mag_norm = sqrt(pow(rho[1][ir], 2) + pow(rho[2][ir], 2) + pow(rho[3][ir], 2));
        rhoout1[ir] = 0.5 * (rho[0][ir] + mag_norm);
        rhoout2[ir] = 0.5 * (rho[0][ir] - mag_norm);
        if(mag_norm > 1e-12)
        {
            mag_part[ir] = rho[1][ir] / mag_norm;
            mag_part[ir + nrxx] = rho[2][ir] / mag_norm;
            mag_part[ir + 2 * nrxx] = rho[3][ir] / mag_norm;
        }
        else
        {
            mag_part[ir] = 0.0;
            mag_part[ir + nrxx] = 0.0;
            mag_part[ir + 2 * nrxx] = 0.0;
        }
    }
    return;
}

template void XC_Functional::grad_wfc<std::complex<double>, base_device::DEVICE_CPU, double>(
    const int ik,
    const double tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
    const std::complex<double>* rhog,
    std::complex<double>* grad);
#if __CUDA || __ROCM
template void XC_Functional::grad_wfc<std::complex<double>, base_device::DEVICE_GPU, double>(
    const int ik,
    const double tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
    const std::complex<double>* rhog,
    std::complex<double>* grad);
#endif // __CUDA || __ROCM