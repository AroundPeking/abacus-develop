// This file contains interface to xc_functional class:
// 1. v_xc : which takes rho as input, and v_xc as output
// 2. v_xc_libxc : which does the same thing as v_xc, but calling libxc
// NOTE : it is only used for nspin = 1 and 2, the nspin = 4 case is treated in v_xc
// 3. v_xc_meta : which takes rho and tau as input, and v_xc as output

#include "xc_functional.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_parameter/parameter.h"
#include <unistd.h>

// [etxc, vtxc, v] = XC_Functional::v_xc(...)
std::tuple<double,double,ModuleBase::matrix> XC_Functional::v_xc(
	const int &nrxx, // number of real-space grid
    const Charge* const chr,
    const UnitCell *ucell) // core charge density
{
    ModuleBase::TITLE("XC_Functional","v_xc");
    ModuleBase::timer::tick("XC_Functional","v_xc");

    if(use_libxc)
    {
#ifdef USE_LIBXC
        return v_xc_libxc(nrxx, ucell->omega, ucell->tpiba, chr);
#else
        ModuleBase::WARNING_QUIT("v_xc","compile with LIBXC");
#endif
    }

    //Exchange-Correlation potential Vxc(r) from n(r)
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(GlobalV::NSPIN, nrxx);

    // the square of the e charge
    // in Rydeberg unit, so * 2.0.
    double e2 = 2.0;

    double vanishing_charge = 1.0e-10;

    if (GlobalV::NSPIN == 1 || ( GlobalV::NSPIN ==4 && !GlobalV::DOMAG && !GlobalV::DOMAG_Z))
    {
        // spin-unpolarized case
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            // total electron charge density
            double rhox = chr->rho[0][ir] + chr->rho_core[ir];
            if(GlobalV::use_paw && rhox < 1e-14) rhox = 1e-14;
            double arhox = std::abs(rhox);
            if (arhox > vanishing_charge)
            {
                double exc = 0.0;
                double vxc = 0.0;
                XC_Functional::xc(arhox, exc, vxc);
                v(0,ir) = e2 * vxc;
                // consider the total charge density
                etxc += e2 * exc * rhox;
                // only consider chr->rho
                vtxc += v(0, ir) * chr->rho[0][ir];
            } // endif
        } //enddo
    }
    else if(GlobalV::NSPIN ==2)
    {
        // spin-polarized case
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            double rhox = chr->rho[0][ir] + chr->rho[1][ir] + chr->rho_core[ir]; //HLX(05-29-06): bug fixed
            double arhox = std::abs(rhox);

            if (arhox > vanishing_charge)
            {
                double zeta = (chr->rho[0][ir] - chr->rho[1][ir]) / arhox; //HLX(05-29-06): bug fixed

                if (std::abs(zeta)  > 1.0)
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }

                double rhoup = arhox * (1.0+zeta) / 2.0;
                double rhodw = arhox * (1.0-zeta) / 2.0;
                double exc = 0.0;
                double vxc[2];
                XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);

                for (int is = 0;is < GlobalV::NSPIN;is++)
                {
                    v(is, ir) = e2 * vxc[is];
                }

                etxc += e2 * exc * rhox;
                vtxc += v(0, ir) * chr->rho[0][ir] + v(1, ir) * chr->rho[1][ir];
            }
        }
    }
    else if(GlobalV::NSPIN == 4)//noncollinear case added by zhengdy
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for(int ir = 0;ir<nrxx; ir++)
        {
            double amag = sqrt( pow(chr->rho[1][ir],2) + pow(chr->rho[2][ir],2) + pow(chr->rho[3][ir],2) );

            double rhox = chr->rho[0][ir] + chr->rho_core[ir];

            double arhox = std::abs( rhox );

            if ( arhox > vanishing_charge )
            {
                double zeta = amag / arhox;
                double exc = 0.0;
                double vxc[2];

                if ( std::abs( zeta ) > 1.0 )
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }//end if

                if(use_libxc)
                {
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional::xc_spin_libxc(rhoup, rhodw, exc, vxc[0], vxc[1]);
                }
                else
                {
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);
                }

                etxc += e2 * exc * rhox;

                v(0, ir) = e2*( 0.5 * ( vxc[0] + vxc[1]) );
                vtxc += v(0,ir) * chr->rho[0][ir];

                double vs = 0.5 * ( vxc[0] - vxc[1] );
                if ( amag > vanishing_charge )
                {
                    for(int ipol = 1;ipol< 4;ipol++)
                    {
                        v(ipol, ir) = e2 * vs * chr->rho[ipol][ir] / amag;
                        vtxc += v(ipol,ir) * chr->rho[ipol][ir];
                    }//end do
                }//end if
            }//end if
        }//end do
    }//end if
    // energy terms, local-density contributions

    // add gradient corrections (if any)
    // mohan modify 2009-12-15

    // the dummy variable dum contains gradient correction to stress
    // which is not used here
    std::vector<double> dum;
    gradcorr(etxc, vtxc, v, chr, chr->rhopw, ucell, dum);

    // parallel code : collect vtxc,etxc
    // mohan add 2008-06-01
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    etxc *= ucell->omega / chr->rhopw->nxyz;
    vtxc *= ucell->omega / chr->rhopw->nxyz;

    ModuleBase::timer::tick("XC_Functional","v_xc");
    return std::make_tuple(etxc, vtxc, std::move(v));
}

#ifdef USE_LIBXC

std::tuple<double,double,ModuleBase::matrix> XC_Functional::v_xc_libxc(		// Peize Lin update for nspin==4 at 2023.01.14
        const int &nrxx, // number of real-space grid
        const double &omega, // volume of cell
        const double tpiba,
        const Charge* const chr)
{
    ModuleBase::TITLE("XC_Functional","v_xc_libxc");
    ModuleBase::timer::tick("XC_Functional","v_xc_libxc");

    const int nspin = 
        (GlobalV::NSPIN == 1 || ( GlobalV::NSPIN ==4 && !GlobalV::DOMAG && !GlobalV::DOMAG_Z))
        ? 1 : 2;

    // variable nspin = 2 for normal noncollinear (nspin = 4) calculation

    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(nspin,nrxx);

    //----------------------------------------------------------
    // xc_func_type is defined in Libxc package
    // to understand the usage of xc_func_type,
    // use can check on website, for example:
    // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/
    //----------------------------------------------------------

    std::vector<xc_func_type> funcs = init_func( (1==nspin) ? XC_UNPOLARIZED:XC_POLARIZED );

    const bool is_gga = [&funcs]()
    {
        for( xc_func_type &func : funcs )
        {
            switch( func.info->family )
            {
                case XC_FAMILY_GGA:
                case XC_FAMILY_HYB_GGA:
                    return true;
            }
        }
        return false;
    }();

    // converting rho
    std::vector<double> rho(nrxx*nspin);
    std::vector<double> amag;
    if(1==nspin || 2==GlobalV::NSPIN)
    {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static, 1024)
        #endif
        for( int is=0; is<nspin; ++is )
            for( int ir=0; ir<nrxx; ++ir )
                rho[ir*nspin+is] = chr->rho[is][ir] + 1.0/nspin*chr->rho_core[ir];
    }
    else // noncollinear
    {
        amag.resize(nrxx);
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for( int ir=0; ir<nrxx; ++ir )
        {
            const double arhox = std::abs( chr->rho[0][ir] + chr->rho_core[ir] );
            amag[ir] = std::sqrt( std::pow(chr->rho[1][ir],2) + std::pow(chr->rho[2][ir],2) + std::pow(chr->rho[3][ir],2) );
            //const double amag_clip = (amag[ir]<arhox) ? amag[ir] : arhox;
            const double amag_clip = amag[ir];
            rho[ir*nspin+0] = (arhox + amag_clip) / 2.0; // n_{+}
            rho[ir*nspin+1] = (arhox - amag_clip) / 2.0; // n_{-}
        }        
    }

    std::vector<std::vector<ModuleBase::Vector3<double>>> gdr;
    std::vector<double> sigma;
    if(is_gga)
    {
        // calculating grho
        gdr.resize( nspin );
        for (int is = 0; is < nspin; is++)
        {
            gdr[is].resize(nrxx);
        }


        // new block
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if (GlobalV::NSPIN == 4 && PARAM.inp.gga_grad == 2)
        {
            // compute mag_part = mag_i / |mag|
            std::vector<double> mag_part(3 * nrxx, 0.0);
            for (int ir = 0; ir < nrxx; ir++)
            {
                double mag_norm = std::sqrt(std::pow(chr->rho[1][ir], 2) + std::pow(chr->rho[2][ir], 2) + std::pow(chr->rho[3][ir], 2));
                if(mag_norm > 1e-12)
                {
                    mag_part[ir] = chr->rho[1][ir] / mag_norm;
                    mag_part[ir + nrxx] = chr->rho[2][ir] / mag_norm;
                    mag_part[ir + 2 * nrxx] = chr->rho[3][ir] / mag_norm;
                }
            }

            // compute \nabla rho' = \nabla rho_core + \nabla rho
            std::complex<double>* rhogsum1 = new std::complex<double>[chr->rhopw->npw];
            const double fac0 = GlobalV::NSPIN == 2 ? 0.5 : 1.0;
            for (int ig = 0; ig < chr->rhopw->npw; ig++)
            {
                rhogsum1[ig] = chr->rhog[0][ig] + fac0 * chr->rhog_core[ig];
            }

            ModuleBase::Vector3<double>* gdr1 = new ModuleBase::Vector3<double>[nrxx];
            XC_Functional::grad_rho(rhogsum1, gdr1, chr->rhopw, tpiba);

            // for non-collinear case
            // rho' has been calculated in rhotmp1, rhog' has been calculated in rhogsum1, \nabla rho' has been calculated
            // in gdr1
            std::complex<double>* tmp_recip = new std::complex<double>[chr->rhopw->npw];
            ModuleBase::Vector3<double>* gdr_mag = new ModuleBase::Vector3<double>[nrxx];
            for (int ir = 0; ir < nrxx; ir++)
            {
                gdr_mag[ir] = gdr1[ir];
                gdr[0][ir] = 0.5 * gdr_mag[ir];
                gdr[1][ir] = 0.5 * gdr_mag[ir];
            }
            // now gdr[0] and gdr[1] are both 0.5 * \nabla rho'

            for (int is = 1; is < 4; is++) // loop over mx, my, mz
            {
                chr->rhopw->real2recip(chr->rho[is], tmp_recip);
                XC_Functional::grad_rho(tmp_recip, gdr_mag, chr->rhopw, tpiba);
                // now gdr_mag contains \nabla mag_i

                // mag_part is mag_i / |mag|
                const double* mag_part_is = mag_part.data() + (is-1) * nrxx;
                for (int ir = 0; ir < nrxx; ir++)
                {
                    const ModuleBase::Vector3<double> grad_is = 0.5 * gdr_mag[ir] * mag_part_is[ir];
                    // grad_is = 0.5 * (\nabla mag_i) * (mag_i / |mag|)
                    gdr[0][ir] += grad_is;
                    gdr[1][ir] -= grad_is;
                }
            }

            delete[] tmp_recip;
            delete[] gdr_mag;
            delete[] gdr1;
            delete[] rhogsum1;
        }
        else
        {
            //<<<<<<<<<<<<<<<<< this block is the original algorithm
            for( int is=0; is!=nspin; ++is )
            {
                std::vector<double> rhor(nrxx);
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static, 1024)
                #endif
                for(int ir=0; ir<nrxx; ++ir)
                    rhor[ir] = rho[ir*nspin+is];
                //------------------------------------------
                // initialize the charge density array in reciprocal space
                // bring electron charge density from real space to reciprocal space
                //------------------------------------------
                std::vector<std::complex<double>> rhog(chr->rhopw->npw);
                chr->rhopw->real2recip(rhor.data(), rhog.data());

                //-------------------------------------------
                // compute the gradient of charge density and
                // store the gradient in gdr[is]
                //-------------------------------------------
                gdr[is].resize(nrxx);
                XC_Functional::grad_rho(rhog.data(), gdr[is].data(), chr->rhopw, tpiba);
            } // end for(is)

            //>>>>>>>>>>>>>>>>> end of the original algorithm
        }
        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


        // gdr stores the gradients of n_{+} and n_{-}

        // converting grho
        sigma.resize( nrxx * ((1==nspin)?1:3) );
        if( 1==nspin )
        {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static, 1024)
            #endif
            for( int ir=0; ir<nrxx; ++ir )
                sigma[ir] = gdr[0][ir]*gdr[0][ir];
        }
        else
        {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static, 256)
            #endif
            for( int ir=0; ir<nrxx; ++ir )
            {
                sigma[ir*3]   = gdr[0][ir]*gdr[0][ir];
                sigma[ir*3+1] = gdr[0][ir]*gdr[1][ir];
                sigma[ir*3+2] = gdr[1][ir]*gdr[1][ir];
            }
        }

        // contracted gradients sigma:
        // sigma0 = \grad n_{+} \cdot \grad n_{+}
        // sigma1 = \grad n_{+} \cdot \grad n_{-}
        // sigma2 = \grad n_{-} \cdot \grad n_{-}


    } // end if(is_gga)

    for( xc_func_type &func : funcs )
    {
        // jiyy add for threshold
        constexpr double rho_threshold = 1E-6;
        constexpr double grho_threshold = 1E-10;

        xc_func_set_dens_threshold(&func, rho_threshold);

        // sgn for threshold mask
        std::vector<double> sgn(nrxx*nspin, 1.0);
        // in the case of GGA correlation for polarized case,
        // a cutoff for grho is required to ensure that libxc gives reasonable results
        if(nspin==2 && func.info->family != XC_FAMILY_LDA && func.info->kind==XC_CORRELATION)
        {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static, 512)
            #endif
            for( int ir=0; ir<nrxx; ++ir )
            {
                if ( rho[ir*2]<rho_threshold || std::sqrt(std::abs(sigma[ir*3]))<grho_threshold )
                    sgn[ir*2] = 0.0;
                if ( rho[ir*2+1]<rho_threshold || std::sqrt(std::abs(sigma[ir*3+2]))<grho_threshold )
                    sgn[ir*2+1] = 0.0;
            }
        }
        // sgn[even] -> mask for n_{+}
        // sgn[odd]  -> mask for n_{-}

        std::vector<double> exc   ( nrxx                    );
        std::vector<double> vrho  ( nrxx * nspin            );
        std::vector<double> vsigma( nrxx * ((1==nspin)?1:3) ); // nrxx * 3 for noncollinear
        switch( func.info->family )
        {
            case XC_FAMILY_LDA:
                // call Libxc function: xc_lda_exc_vxc
                xc_lda_exc_vxc( &func, nrxx, rho.data(), 
                    exc.data(), vrho.data() );
                break;
            case XC_FAMILY_GGA:
            case XC_FAMILY_HYB_GGA:
                // call Libxc function: xc_gga_exc_vxc
                xc_gga_exc_vxc( &func, nrxx, rho.data(), sigma.data(),
                    exc.data(), vrho.data(), vsigma.data() );
                break;
            default:
                throw std::domain_error("func.info->family ="+std::to_string(func.info->family)
                    +" unfinished in "+std::string(__FILE__)+" line "+std::to_string(__LINE__));
                break;
        }

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:etxc) schedule(static, 256)
        #endif
        for( int is=0; is<nspin; ++is )
            for( int ir=0; ir< nrxx; ++ir )
                etxc += ModuleBase::e2 * exc[ir] * rho[ir*nspin+is] * sgn[ir*nspin+is];

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:vtxc) schedule(static, 256)
        #endif
        for( int is=0; is<nspin; ++is )
        {
            for( int ir=0; ir< nrxx; ++ir )
            {
                const double v_tmp = ModuleBase::e2 * vrho[ir*nspin+is] * sgn[ir*nspin+is];
                v(is,ir) += v_tmp;
                vtxc += v_tmp * rho[ir*nspin+is];
            }
        }

        if(func.info->family == XC_FAMILY_GGA || func.info->family == XC_FAMILY_HYB_GGA)
        {
            std::vector<std::vector<ModuleBase::Vector3<double>>> h( nspin,
                std::vector<ModuleBase::Vector3<double>>(nrxx) );
            if( 1==nspin )
            {
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static, 1024)
                #endif
                for( int ir=0; ir< nrxx; ++ir )
                    h[0][ir] = 2.0 * gdr[0][ir] * vsigma[ir] * 2.0 * sgn[ir];
            }
            else
            {
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static, 1024)
                #endif
                for( int ir=0; ir< nrxx; ++ir )
                {
                    h[0][ir] = 2.0 * (gdr[0][ir] * vsigma[ir*3  ] * sgn[ir*2  ] * 2.0 
                                    + gdr[1][ir] * vsigma[ir*3+1] * sgn[ir*2]   * sgn[ir*2+1]);
                    h[1][ir] = 2.0 * (gdr[1][ir] * vsigma[ir*3+2] * sgn[ir*2+1] * 2.0 
                                    + gdr[0][ir] * vsigma[ir*3+1] * sgn[ir*2]   * sgn[ir*2+1]);
                }

                // de/d(\nabla n_{+}) = 2 * \nabla n_{+} * vsigma0 + \nabla n_{-} * vsigma1
                // de/d(\nabla n_{-}) = 2 * \nabla n_{-} * vsigma2 + \nabla n_{+} * vsigma1
                //
                // NOTE: the mask for vsigma0/vsigma2 is simply the mask for n_{+}/n_{-},
                // but the mask for vsigma1 is taken to be the product of masks for n_{+} and n_{-}
            }

            // define two dimensional array dh [ nspin, nrxx ]
            std::vector<std::vector<double>> dh(nspin, std::vector<double>(nrxx));
            for( int is=0; is!=nspin; ++is )
                XC_Functional::grad_dot( h[is].data(), dh[is].data(), chr->rhopw, tpiba);

            // dh is \div h

            double rvtxc = 0.0;
            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) reduction(+:rvtxc) schedule(static, 256)
            #endif
            for( int is=0; is<nspin; ++is )
            {
                for( int ir=0; ir< nrxx; ++ir )
                {
                    rvtxc += dh[is][ir] * rho[ir*nspin+is];
                    v(is,ir) -= dh[is][ir];
                }
            }

            vtxc -= rvtxc;
        } // end if(func.info->family == XC_FAMILY_GGA || func.info->family == XC_FAMILY_HYB_GGA))
    } // end for( xc_func_type &func : funcs )

    //-------------------------------------------------
    // for MPI, reduce the exchange-correlation energy
    //-------------------------------------------------
    #ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
    #endif

    etxc *= omega / chr->rhopw->nxyz;
    vtxc *= omega / chr->rhopw->nxyz;

    finish_func(funcs);

    if(1==GlobalV::NSPIN || 2==GlobalV::NSPIN)
    {
        ModuleBase::timer::tick("XC_Functional","v_xc_libxc");
        return std::make_tuple( etxc, vtxc, std::move(v) );
    }
    else if(4==GlobalV::NSPIN)
    {
        constexpr double vanishing_charge = 1.0e-10;
        ModuleBase::matrix v_nspin4(GlobalV::NSPIN,nrxx);
        for( int ir=0; ir<nrxx; ++ir )
            v_nspin4(0,ir) = 0.5 * (v(0,ir)+v(1,ir));

        // v_nspin4(0) is 0.5 * (v_{+} + v_{-})

        if(GlobalV::DOMAG || GlobalV::DOMAG_Z)
        {
            for( int ir=0; ir<nrxx; ++ir )
            {
                if ( amag[ir] > vanishing_charge )
                {
                    const double vs = 0.5 * (v(0,ir)-v(1,ir));
                    for(int ipol=1; ipol<4; ++ipol)
                        v_nspin4(ipol,ir) = vs * chr->rho[ipol][ir] / amag[ir];
                }
            }
        // v_nspin4(i) is 0.5 * (v_{+} - v_{-}) * mag_i / |mag|
        }

        ModuleBase::timer::tick("XC_Functional","v_xc_libxc");
        return std::make_tuple( etxc, vtxc, std::move(v_nspin4) );
    } // end if(4==GlobalV::NSPIN)
    else//NSPIN != 1,2,4 is not supported
    {
        throw std::domain_error("GlobalV::NSPIN ="+std::to_string(GlobalV::NSPIN)
            +" unfinished in "+std::string(__FILE__)+" line "+std::to_string(__LINE__));
    }
}

//the interface to libxc xc_mgga_exc_vxc(xc_func,n,rho,grho,laplrho,tau,e,v1,v2,v3,v4)
//xc_func : LIBXC data type, contains information on xc functional
//n: size of array, nspin*nnr
//rho,grho,laplrho: electron density, its gradient and laplacian
//tau(kin_r): kinetic energy density
//e: energy density
//v1-v4: derivative of energy density w.r.t rho, gradient, laplacian and tau
//v1 and v2 are combined to give v; v4 goes into vofk

//XC_POLARIZED, XC_UNPOLARIZED: internal flags used in LIBXC, denote the polarized(nspin=1) or unpolarized(nspin=2) calculations, definition can be found in xc.h from LIBXC

// [etxc, vtxc, v, vofk] = XC_Functional::v_xc(...)
std::tuple<double,double,ModuleBase::matrix,ModuleBase::matrix> XC_Functional::v_xc_meta(
    const int &nrxx, // number of real-space grid
    const double &omega, // volume of cell
    const double tpiba,
    const Charge* const chr)
{
    ModuleBase::TITLE("XC_Functional","v_xc_meta");
    ModuleBase::timer::tick("XC_Functional","v_xc_meta");

    double e2 = 2.0;

    //output of the subroutine
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(GlobalV::NSPIN,nrxx);
    ModuleBase::matrix vofk(GlobalV::NSPIN,nrxx);

    //----------------------------------------------------------
    // xc_func_type is defined in Libxc package
    // to understand the usage of xc_func_type,
    // use can check on website, for example:
    // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/
    //----------------------------------------------------------
    
    const int nspin = GlobalV::NSPIN;
    std::vector<xc_func_type> funcs = init_func( ( (1==nspin) ? XC_UNPOLARIZED:XC_POLARIZED ) );

    // converting rho
    std::vector<double> rho;
    rho.resize(nrxx*nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for( int is=0; is<nspin; ++is )
    {
        for( int ir=0; ir<nrxx; ++ir )
        {
            rho[ir*nspin+is] = chr->rho[is][ir] + 1.0/nspin*chr->rho_core[ir];
        }
    }

    std::vector<std::vector<ModuleBase::Vector3<double>>> gdr;
    std::vector<double> sigma;

    // calculating grho
    gdr.resize( nspin );
    for( int is=0; is!=nspin; ++is )
    {
        std::vector<double> rhor(nrxx);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir=0; ir<nrxx; ++ir)
        {
            rhor[ir] = rho[ir*nspin+is];
        }
        //------------------------------------------
        // initialize the charge density array in reciprocal space
        // bring electron charge density from real space to reciprocal space
        //------------------------------------------
        std::vector<std::complex<double>> rhog(chr->rhopw->npw);
        chr->rhopw->real2recip(rhor.data(), rhog.data());

        //-------------------------------------------
        // compute the gradient of charge density and
        // store the gradient in gdr[is]
        //-------------------------------------------
        gdr[is].resize(nrxx);
        XC_Functional::grad_rho(rhog.data(), gdr[is].data(), chr->rhopw, tpiba);
    }

    // converting grho
    sigma.resize( nrxx * ((1==nspin)?1:3) );

    if( 1==nspin )
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for( int ir=0; ir<nrxx; ++ir )
        {
            sigma[ir] = gdr[0][ir]*gdr[0][ir];
        } 
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
        for( int ir=0; ir<nrxx; ++ir )
        {
            sigma[ir*3]   = gdr[0][ir]*gdr[0][ir];
            sigma[ir*3+1] = gdr[0][ir]*gdr[1][ir];
            sigma[ir*3+2] = gdr[1][ir]*gdr[1][ir];
        }
    }

    //converting kin_r
    std::vector<double> kin_r;
    kin_r.resize(nrxx*nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for( int is=0; is<nspin; ++is )
    {
        for( int ir=0; ir<nrxx; ++ir )
        {
            kin_r[ir*nspin+is] = chr->kin_r[is][ir] / 2.0;
        }
    }

    std::vector<double> exc    ( nrxx                    );
    std::vector<double> vrho   ( nrxx * nspin            );
    std::vector<double> vsigma ( nrxx * ((1==nspin)?1:3) );
    std::vector<double> vtau   ( nrxx * nspin            );
    std::vector<double> vlapl  ( nrxx * nspin            );

    const double rho_th  = 1e-8;
    const double grho_th = 1e-12;
    const double tau_th  = 1e-8;
    // sgn for threshold mask
    std::vector<double> sgn( nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for(int i = 0; i < nrxx * nspin; ++i)
    {
        sgn[i] = 1.0;
    }

    if(nspin == 1)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for( int ir=0; ir<nrxx; ++ir )
        {
            if ( rho[ir]<rho_th || sqrt(std::abs(sigma[ir]))<grho_th || std::abs(kin_r[ir])<tau_th)
            {
                sgn[ir] = 0.0;
            }
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for( int ir=0; ir<nrxx; ++ir )
        {
            if ( rho[ir*2]<rho_th || sqrt(std::abs(sigma[ir*3]))<grho_th || std::abs(kin_r[ir*2])<tau_th)
                sgn[ir*2] = 0.0;
            if ( rho[ir*2+1]<rho_th || sqrt(std::abs(sigma[ir*3+2]))<grho_th || std::abs(kin_r[ir*2+1])<tau_th)
                sgn[ir*2+1] = 0.0;
        }
    }

    for ( xc_func_type &func : funcs )
    {
        assert(func.info->family == XC_FAMILY_MGGA);
        xc_mgga_exc_vxc(&func, nrxx, rho.data(), sigma.data(), sigma.data(),
            kin_r.data(), exc.data(), vrho.data(), vsigma.data(), vlapl.data(), vtau.data());

        //process etxc
        for( int is=0; is!=nspin; ++is )
        {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) schedule(static, 256)
#endif
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && get_func_type() == 5)
                {
                    exc[ir] *= (1.0 - XC_Functional::hybrid_alpha);
                }
#endif
                etxc += ModuleBase::e2 * exc[ir] * rho[ir*nspin+is]  * sgn[ir*nspin+is];
            }   
        }

        //process vtxc
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:vtxc) schedule(static, 256)
#endif
        for( int is=0; is<nspin; ++is )
        {
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && get_func_type() == 5)
                {
                    vrho[ir*nspin+is] *= (1.0 - XC_Functional::hybrid_alpha);
                }
#endif
                const double v_tmp = ModuleBase::e2 * vrho[ir*nspin+is]  * sgn[ir*nspin+is];
                v(is,ir) += v_tmp;
                vtxc += v_tmp * chr->rho[is][ir];
            }
        }

        //process vsigma
        std::vector<std::vector<ModuleBase::Vector3<double>>> h( nspin, std::vector<ModuleBase::Vector3<double>>(nrxx) );
        if( 1==nspin )
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && get_func_type() == 5)
                {
                    vsigma[ir] *= (1.0 - XC_Functional::hybrid_alpha);
                }
#endif
                h[0][ir] = 2.0 * gdr[0][ir] * vsigma[ir] * 2.0 * sgn[ir];
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 64)
#endif
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && get_func_type() == 5)
                {
                    vsigma[ir*3]   *= (1.0 - XC_Functional::hybrid_alpha);
                    vsigma[ir*3+1] *= (1.0 - XC_Functional::hybrid_alpha);
                    vsigma[ir*3+2] *= (1.0 - XC_Functional::hybrid_alpha);
                }
#endif
                h[0][ir] = 2.0 * (gdr[0][ir] * vsigma[ir*3  ] * sgn[ir*2  ] * 2.0 
                                + gdr[1][ir] * vsigma[ir*3+1] * sgn[ir*2]   * sgn[ir*2+1]);
                h[1][ir] = 2.0 * (gdr[1][ir] * vsigma[ir*3+2] * sgn[ir*2+1] * 2.0 
                                + gdr[0][ir] * vsigma[ir*3+1] * sgn[ir*2]   * sgn[ir*2+1]);
            }
        }

        // define two dimensional array dh [ nspin, nrxx ]
        std::vector<std::vector<double>> dh(nspin, std::vector<double>( nrxx));
        for( int is=0; is!=nspin; ++is )
        {
            XC_Functional::grad_dot( ModuleBase::GlobalFunc::VECTOR_TO_PTR(h[is]),
                ModuleBase::GlobalFunc::VECTOR_TO_PTR(dh[is]), chr->rhopw,
                tpiba);
        }

        double rvtxc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:rvtxc) schedule(static, 256)
#endif
        for( int is=0; is<nspin; ++is )
        {
            for( int ir=0; ir< nrxx; ++ir )
            {
                rvtxc += dh[is][ir] * rho[ir*nspin+is];
                v(is,ir) -= dh[is][ir];
            }
        }
        vtxc -= rvtxc;

        //process vtau
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for( int is=0; is<nspin; ++is )
        {
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && get_func_type() == 5)
                {
                    vtau[ir*nspin+is] *= (1.0 - XC_Functional::hybrid_alpha);
                }
#endif
                vofk(is,ir) += vtau[ir*nspin+is]  * sgn[ir*nspin+is];
            }
        }
    }
        
    //-------------------------------------------------
    // for MPI, reduce the exchange-correlation energy
    //-------------------------------------------------
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif

    etxc *= omega / chr->rhopw->nxyz;
    vtxc *= omega / chr->rhopw->nxyz;

    finish_func(funcs);

    ModuleBase::timer::tick("XC_Functional","v_xc_meta");
    return std::make_tuple( etxc, vtxc, std::move(v), std::move(vofk) );

}

#endif    //ifdef USE_LIBXC
