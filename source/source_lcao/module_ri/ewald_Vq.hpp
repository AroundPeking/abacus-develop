//=======================
// AUTHOR : jiyy
// DATE :   2023-12-08
//=======================

#ifndef EWALD_VQ_HPP
#define EWALD_VQ_HPP

#include "ewald_Vq.h"
#include "exx_abfs-construct_orbs.h"
#include "module_base/tool_title.h"
#include "module_base/timer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/wavefunc_in_pw.h"
#include "module_base/realarray.h"
#include "module_base/math_ylmreal.h"
#include "module_base/math_polyint.h"
#include <omp.h>
#include <cmath>

// Zc
template<typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq1(
                        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
                        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_ccp,
                        const K_Vectors* kv,
						const ModulePW::PW_Basis_K* wfc_basis,
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						const double& gk_ecut)
-> std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
{
	ModuleBase::TITLE("Ewald_Vq","cal_Vq1");
    ModuleBase::timer::tick("Ewald_Vq","cal_Vq1");

	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>> datas;
	datas.resize(nks0);
	std::map<int, int> abfs_nw = Exx_Abfs::Construct_Orbs::get_nw(abfs);
	std::map<int, int> abfs_ccp_nw = Exx_Abfs::Construct_Orbs::get_nw(abfs_ccp);
	std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>, std::vector<std::vector<ModuleBase::ComplexMatrix>>> result1 = this->get_orb_q(kv, wfc_basis, abfs, gk_ecut);
	std::vector<std::vector<ModuleBase::Vector3<double>>> gks = result1.first;
	std::vector<std::vector<ModuleBase::ComplexMatrix>> abfs_in_Gs = result1.second;

	std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>, std::vector<std::vector<ModuleBase::ComplexMatrix>>> result2 = this->get_orb_q(kv, wfc_basis, abfs_ccp, gk_ecut);
	std::vector<std::vector<ModuleBase::ComplexMatrix>> abfs_ccp_in_Gs = result2.second;

	std::set<TA> unique_set_A1;
	for(const auto& pair : list_A1)
		unique_set_A1.insert(pair.first);
	std::vector<TA> unique_list_A1(unique_set_A1.begin(), unique_set_A1.end());

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		const int npw = gks[ik].size();

		//#pragma omp parallel
		for(size_t i0=0; i0!=list_A0.size(); ++i0)
		{
			//#pragma omp for schedule(dynamic) nowait
			for(size_t i1=0; i1!=unique_list_A1.size(); ++i1)
			{
				const TA iat0 = list_A0[i0];
				const int it0 = GlobalC::ucell.iat2it[iat0];
				const int ia0 = GlobalC::ucell.iat2ia[iat0];
				const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];

				const TA iat1 = unique_list_A1[i1];
				const int ia1 = GlobalC::ucell.iat2ia[iat1];
				const int it1 = GlobalC::ucell.iat2it[iat1];
				const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];
				
				const size_t abfs_ccp_nw_t = abfs_ccp_nw[it0];
				const size_t abfs_nw_t = abfs_nw[it1];
                RI::Tensor<std::complex<double>> data({abfs_ccp_nw_t, abfs_nw_t});

				for(size_t iw0=0; iw0!=abfs_ccp_nw_t; ++iw0)
				{
					for(size_t iw1=0; iw1!=abfs_nw_t; ++iw1)
					{
						for(size_t ig=0; ig!=npw; ++ig)
						{
							std::complex<double> phase = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT*(gks[ik][ig]*(tau0-tau1)));
							data(iw0, iw1) += std::conj(abfs_ccp_in_Gs[ik][it0](iw0, ig)) * abfs_in_Gs[ik][it1](iw1, ig) * phase; //* vg[ig];
						}
					}
				}
				//#pragma omp critical(Ewald_Vq_cal_Vq1)
				datas[ik][iat0][iat1] = data;
			}	
		}
	}

    ModuleBase::timer::tick("Ewald_Vq","cal_Vq1");
	return datas;
}

template<typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vq2(
	const K_Vectors* kv, 
	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs)
-> std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
{
	ModuleBase::TITLE("Ewald_Vq","cal_Vq2");
    ModuleBase::timer::tick("Ewald_Vq","cal_Vq2");

	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>> datas;
	datas.resize(nks0);

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		for(const auto &Vs_tmpA : Vs)
		{
			const TA &iat0 = Vs_tmpA.first;
			for(const auto &Vs_tmpB : Vs_tmpA.second)
			{
				const TA &iat1 = Vs_tmpB.first.first;
				const TC &cell1 = Vs_tmpB.first.second;
				std::complex<double> phase = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1)*GlobalC::ucell.latvec)));
				if(datas[ik][iat0][iat1].empty())
					datas[ik][iat0][iat1] = RI::Global_Func::convert<std::complex<double>>(Vs[iat0][Vs_tmpB.first]) * phase;
				else
					datas[ik][iat0][iat1] = datas[ik][iat0][iat1] + RI::Global_Func::convert<std::complex<double>>(Vs[iat0][Vs_tmpB.first]) * phase;
			}
		}
	}

    ModuleBase::timer::tick("Ewald_Vq","cal_Vq2");
	return datas;
}

template<typename Tdata>
bool 
Ewald_Vq<Tdata>::check_Vs(
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs2,
                        const double &ccp_rmesh_times)
{
	for(size_t i0=0; i0!=list_A0.size(); ++i0)
	{
		for(size_t i1=0; i1!=list_A1.size(); ++i1)
		{
			const TA iat0 = list_A0[i0];
			const TA iat1 = list_A1[i1].first;
			const TC &cell1 = list_A1[i1].second;
			const int it0 = GlobalC::ucell.iat2it[iat0];
			const int ia0 = GlobalC::ucell.iat2ia[iat0];
			const int it1 = GlobalC::ucell.iat2it[iat1];
			const int ia1 = GlobalC::ucell.iat2ia[iat1];
			const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];
			const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];
			const double Rcut = std::min(
				GlobalC::ORB.Phi[it0].getRcut() * ccp_rmesh_times + GlobalC::ORB.Phi[it1].getRcut(),
				GlobalC::ORB.Phi[it1].getRcut() * ccp_rmesh_times + GlobalC::ORB.Phi[it0].getRcut());
			const Abfs::Vector3_Order<double> R_delta = -tau0+tau1+(RI_Util::array3_to_Vector3(cell1)*GlobalC::ucell.latvec);
			if( R_delta.norm()*GlobalC::ucell.lat0 < Rcut )
			{
				std::cout<<"iat0: "<<iat0<<"\tiat1: "<<iat1<<"\tcell: ("<<cell1[0]<<", "<<cell1[1]<<", "<<cell1[2]<<")\tSR1: "<<Vs1[list_A0[i0]][list_A1[i1]](0, 0)<<"\tSR_ori: "<<Vs2[list_A0[i0]][list_A1[i1]](0, 0)<<std::endl;
			}
		}
	}
	return true;
}

template<typename Tdata>
bool
Ewald_Vq<Tdata>::check_Vq(
	const K_Vectors* kv, 
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>& Vq1,
	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>& Vq2
)
{
	ModuleBase::TITLE("Ewald_Vq","check_Vq");

	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>> datas;
	datas.resize(nks0);

	std::set<TA> unique_set_A1;
	for(const auto& pair : list_A1)
		unique_set_A1.insert(pair.first);
	std::vector<TA> unique_list_A1(unique_set_A1.begin(), unique_set_A1.end());

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		for(size_t i0=0; i0!=list_A0.size(); ++i0)
		{
			for(size_t i1=0; i1!=unique_list_A1.size(); ++i1)
			{
				const TA iat0 = list_A0[i0];
				const TA iat1 = unique_list_A1[i1];
				std::cout<<"ik: "<<ik<<"\tiat0: "<<iat0<<"\tiat1: "<<iat1<<"\tVq1: "<<Vq1[ik][iat0][iat1](0, 0)<<"\tVq2: "<<Vq2[ik][iat0][iat1](0, 0)<<std::endl;
			}
		}
	}
	return true;
}

template<typename Tdata>
std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>, std::vector<std::vector<ModuleBase::ComplexMatrix>>>
Ewald_Vq<Tdata>::get_orb_q(const K_Vectors* kv, 
						   const ModulePW::PW_Basis_K* wfc_basis,
						   const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in,
						   const double& gk_ecut
						   )
{
	ModuleBase::TITLE("Ewald_Vq","get_orb_q");
    ModuleBase::timer::tick("Ewald_Vq","get_orb_q");

	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	int nmax_total = Exx_Abfs::Construct_Orbs::get_nmax_total(orb_in);
	const int ntype = orb_in.size();
	ModuleBase::realArray table_local(ntype, nmax_total, GlobalV::NQX);
	Wavefunc_in_pw::make_table_q(orb_in, table_local); 

	std::vector<std::vector<ModuleBase::ComplexMatrix>> orb_in_Gs(nks0);
	std::vector<int> npw = this->get_npwk(kv, wfc_basis, gk_ecut);
	std::vector<std::vector<ModuleBase::Vector3<double>>> gks(nks0);

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		gks[ik].resize(npw[ik]);
		for(size_t ig=0; ig!=npw[ik]; ++ig)
			gks[ik][ig] = kv->kvec_c[ik] - this->get_gcar(wfc_basis, ig);

		orb_in_Gs[ik] = this->produce_local_basis_in_pw(ik, gks[ik], orb_in, kv, wfc_basis, table_local, gk_ecut);
	}

	std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>, std::vector<std::vector<ModuleBase::ComplexMatrix>>> result = std::make_pair(gks, orb_in_Gs);

	ModuleBase::timer::tick("Ewald_Vq","get_orb_q");	
	return result;
}

template<typename Tdata>
auto Ewald_Vq<Tdata>::cal_Vs_ewald(const K_Vectors* kv, 
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						std::vector<std::map<TA,std::map<TA, RI::Tensor<std::complex<double>>>>>& Vq,
                        const double &ccp_rmesh_times)
-> std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
{
	ModuleBase::TITLE("Ewald_Vq","cal_Vs_ewald");
    ModuleBase::timer::tick("Ewald_Vq","cal_Vs_ewald");

	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> datas;
	const double SPIN_multiple = std::map<int,double>{{1,0.5}, {2,1}, {4,1}}.at(GlobalV::NSPIN);
	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	
	for(size_t i0=0; i0!=list_A0.size(); ++i0)
	{
		const TA iat0 = list_A0[i0];
		for(size_t i1=0; i1!=list_A1.size(); ++i1)
		{
			const TA iat1 = list_A1[i1].first;
			const TC &cell1 = list_A1[i1].second;
			const int it0 = GlobalC::ucell.iat2it[iat0];
			const int ia0 = GlobalC::ucell.iat2ia[iat0];
			const int it1 = GlobalC::ucell.iat2it[iat1];
			const int ia1 = GlobalC::ucell.iat2ia[iat1];
			const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];
			const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];
			const double Rcut = std::min(
				GlobalC::ORB.Phi[it0].getRcut() * ccp_rmesh_times + GlobalC::ORB.Phi[it1].getRcut(),
				GlobalC::ORB.Phi[it1].getRcut() * ccp_rmesh_times + GlobalC::ORB.Phi[it0].getRcut());
			const Abfs::Vector3_Order<double> R_delta = -tau0+tau1+(RI_Util::array3_to_Vector3(cell1)*GlobalC::ucell.latvec);
			if( R_delta.norm()*GlobalC::ucell.lat0 < Rcut )
			{
				for(size_t ik=0; ik!=nks0; ++ik)
				{
					const std::complex<double> frac = std::exp(- ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(cell1)*GlobalC::ucell.latvec) )) * kv->wk[ik] * SPIN_multiple;
					RI::Tensor<Tdata> Vs_tmp;
					if (static_cast<int>(std::round(SPIN_multiple * kv->wk[ik] * kv->nkstot_full)) == 2)
						Vs_tmp = RI_2D_Comm::tensor_real(RI::Global_Func::convert<Tdata>(Vq[ik][iat0][iat1] * frac));
                	else Vs_tmp = RI::Global_Func::convert<Tdata>(Vq[ik][iat0][iat1] * frac);
					
					if(datas[list_A0[i0]][list_A1[i1]].empty())
						datas[list_A0[i0]][list_A1[i1]] = Vs_tmp;
					else
						datas[list_A0[i0]][list_A1[i1]] = datas[list_A0[i0]][list_A1[i1]] + Vs_tmp;
				}
			}
		}
	}

    ModuleBase::timer::tick("Ewald_Vq","cal_Vs_ewald");
	return datas;
}

template<typename Tdata>
std::vector<ModuleBase::ComplexMatrix>
Ewald_Vq<Tdata>::produce_local_basis_in_pw(const int& ik,
										std::vector<ModuleBase::Vector3<double>>& gk,
										const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in,
										const K_Vectors* kv,
                                        const ModulePW::PW_Basis_K* wfc_basis,
                                        const ModuleBase::realArray& table_local,
										const double& gk_ecut)
{
	ModuleBase::TITLE("Ewald_Vq","produce_local_basis_in_pw");
    ModuleBase::timer::tick("Ewald_Vq","produce_local_basis_in_pw");

	const int npw = gk.size();
	const int ntype = orb_in.size();
	std::map<int, int> orb_nw = Exx_Abfs::Construct_Orbs::get_nw(orb_in);
	std::vector<ModuleBase::ComplexMatrix> psi; 
	psi.resize(ntype);

	int lmax = std::numeric_limits<int>::min();
	for(const auto &out_vec: orb_in)
	{
		for(const auto &value : out_vec)
		{
			int temp = value.size(); 
			if(temp > lmax) 
				lmax = temp;
		}
	}

	const int total_lm = (lmax + 1) * (lmax + 1);
	ModuleBase::matrix ylm(total_lm, npw);

	ModuleBase::YlmReal::Ylm_Real(total_lm, npw, gk.data(), ylm);

	std::vector<double> flq(npw);
	for(size_t T=0; T!=ntype; ++T)
	{
		ModuleBase::ComplexMatrix sub_psi(orb_nw[T], npw);
		int iwall=0;
		int ic = 0;
		for(size_t L=0; L!=orb_in[T].size(); ++L)
		{
			std::complex<double> lphase = pow(ModuleBase::NEG_IMAG_UNIT, L);
			for(size_t N=0; N!=orb_in[T][L].size(); ++N )
			{
				for(size_t ig = 0; ig != npw; ++ig)
					flq[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(table_local,
                                                                        T,
                                                                        ic,
                                                                        GlobalV::NQX,
                                                                        GlobalV::DQ,
                                                                        gk[ig].norm() * GlobalC::ucell.tpiba);

				for(size_t m=0; m!=2*L+1; ++m)
				{
					const int lm = L * L + m;
					for(size_t ig = 0; ig != npw; ++ig)
						sub_psi(iwall, ig) = lphase * ylm(lm, ig) * flq[ig];

					++iwall;
				}
				++ic;
			}// end for N
		}// end for L
		psi[T] = sub_psi;
	} // end for T

    ModuleBase::timer::tick("Ewald_Vq","produce_local_basis_in_pw");
	return psi;
}

template<typename Tdata>
std::vector<int>
Ewald_Vq<Tdata>::get_npwk(const K_Vectors* kv, const ModulePW::PW_Basis_K* wfc_basis, const double& gk_ecut)
{

	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	std::vector<int> npwk(nks0);

	for(size_t ik = 0; ik != nks0; ++ik)
	{
		int ng = 0;
		for(size_t ig = 0; ig != wfc_basis->npw ; ++ig)
		{
			const double gk2 = (this->get_gcar(wfc_basis, ig)+kv->kvec_c[ik]).norm2();
			if (gk2 <= gk_ecut/wfc_basis->tpiba2)
				++ng;
		}
		npwk[ik] = ng;
	}

	return npwk;
}

template<typename Tdata>
ModuleBase::Vector3<double> 
Ewald_Vq<Tdata>::get_gcar(const ModulePW::PW_Basis_K* wfc_basis, const int ig)
{
    int isz = wfc_basis->ig2isz[ig];
    int iz = isz % wfc_basis->nz;
    int is = isz / wfc_basis->nz;
    int ix = wfc_basis->is2fftixy[is] / wfc_basis->fftny;
    int iy = wfc_basis->is2fftixy[is] % wfc_basis->fftny;
    if (ix >= int(wfc_basis->nx/2) + 1) ix -= wfc_basis->nx;
    if (iy >= int(wfc_basis->ny/2) + 1) iy -= wfc_basis->ny;
    if (iz >= int(wfc_basis->nz/2) + 1) iz -= wfc_basis->nz;
    ModuleBase::Vector3<double> f;
    f.x = ix;
    f.y = iy;
    f.z = iz;
    f = f * wfc_basis->G;
    ModuleBase::Vector3<double> g_temp_ = f;
    return g_temp_;
}

#endif
