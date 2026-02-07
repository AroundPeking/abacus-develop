// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#ifndef LRI_CV_HPP
#define LRI_CV_HPP

#include "LRI_CV.h"
#include "LRI_CV_Tools.h"
#include "Inverse_Matrix.h"
#include "module_ri/exx_abfs-abfs-index.h"
#include "exx_abfs-construct_orbs.h"
#include "RI_Util.h"
#include "RI_2D_Comm.h"
#include "module_base/tool_title.h"
#include "module_base/timer.h"
#include <RI/global/Global_Func-1.h>
#include "module_hamilt_lcao/hamilt_lcaodft/wavefunc_in_pw.h"
#include "module_base/realarray.h"
#include <omp.h>
#include<algorithm>

template<typename Tdata>
LRI_CV<Tdata>::LRI_CV()
{
	pthread_rwlock_init(&rwlock_Vw,NULL);
	pthread_rwlock_init(&rwlock_Cw,NULL);
	pthread_rwlock_init(&rwlock_dVw,NULL);
	pthread_rwlock_init(&rwlock_dCw,NULL);
}

template<typename Tdata>
LRI_CV<Tdata>::~LRI_CV()
{
	pthread_rwlock_destroy(&rwlock_Vw);
	pthread_rwlock_destroy(&rwlock_Cw);
	pthread_rwlock_destroy(&rwlock_dVw);
	pthread_rwlock_destroy(&rwlock_dCw);
}


template<typename Tdata>
void LRI_CV<Tdata>::set_orbitals(
	const UnitCell &ucell,
	const LCAO_Orbitals& orb,
	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos_in,
	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs_in,
	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs_ccp_in,
	const double &kmesh_times,
	ORB_gaunt_table& MGT,
    const bool& init_MGT,
    const bool& init_C)
{
	ModuleBase::TITLE("LRI_CV", "set_orbitals");
	ModuleBase::timer::tick("LRI_CV", "set_orbitals");

	this->lcaos = lcaos_in;
	this->abfs = abfs_in;
	this->abfs_ccp = abfs_ccp_in;

	this->lcaos_rcut = Exx_Abfs::Construct_Orbs::get_Rcut(this->lcaos);
    this->abfs_ccp_rcut = Exx_Abfs::Construct_Orbs::get_Rcut(this->abfs_ccp);
    const double lcaos_rmax = Exx_Abfs::Construct_Orbs::get_Rmax(this->lcaos);
    const double abfs_ccp_rmax
        = Exx_Abfs::Construct_Orbs::get_Rmax(this->abfs_ccp);

	const ModuleBase::Element_Basis_Index::Range
		range_lcaos = Exx_Abfs::Abfs_Index::construct_range( lcaos );
	this->index_lcaos = ModuleBase::Element_Basis_Index::construct_index( range_lcaos );

	const ModuleBase::Element_Basis_Index::Range
		range_abfs = Exx_Abfs::Abfs_Index::construct_range( abfs );
	this->index_abfs = ModuleBase::Element_Basis_Index::construct_index( range_abfs );

    int Lmax_v = std::numeric_limits<double>::min();
    this->m_abfs_abfs.init(2, ucell, orb, kmesh_times, lcaos_rmax + abfs_ccp_rmax, Lmax_v);
    int Lmax_c = std::numeric_limits<double>::min();
    if (init_C)
        this->m_abfslcaos_lcaos.init(1, ucell, orb, kmesh_times, lcaos_rmax, Lmax_c);
    int Lmax = std::max(Lmax_v, Lmax_c);

    if (init_MGT) {
        MGT.init_Gaunt_CH(Lmax);
        MGT.init_Gaunt(Lmax);
    }

    this->m_abfs_abfs.init_radial(this->abfs_ccp, this->abfs, MGT);
    this->m_abfs_abfs.init_radial_table();
    if (init_C) {
        this->m_abfslcaos_lcaos.init_radial(this->abfs_ccp,
                                            this->lcaos,
                                            this->lcaos,
                                            MGT);
        this->m_abfslcaos_lcaos.init_radial_table();
    }

	ModuleBase::timer::tick("LRI_CV", "set_orbitals");
}

template <typename Tdata>
double LRI_CV<Tdata>::cal_V_Rcut(const int it0, const int it1) {
    return this->abfs_ccp_rcut[it0] + this->lcaos_rcut[it1];
}

template <typename Tdata>
double LRI_CV<Tdata>::cal_C_Rcut(const int it0, const int it1) {
    return std::min(this->abfs_ccp_rcut[it0], this->lcaos_rcut[it0])
           + this->lcaos_rcut[it1];
}

template<typename Tdata> template<typename Tresult>
auto LRI_CV<Tdata>::cal_datas(
	const UnitCell &ucell,
    const std::vector<TA>& list_A0,
    const std::vector<TAC>& list_A1,
    const std::map<std::string, bool>& flags,
    const T_func_cal_Rcut& func_cal_Rcut,
    const T_func_DPcal_data<Tresult>& func_DPcal_data)
-> std::map<TA,std::map<TAC,Tresult>>
{
	ModuleBase::TITLE("LRI_CV","cal_datas");
	ModuleBase::timer::tick("LRI_CV", "cal_datas");

	std::map<TA,std::map<TAC,Tresult>> Datas;
	#pragma omp parallel
	for(size_t i0=0; i0<list_A0.size(); ++i0)
	{
		#pragma omp for schedule(dynamic) nowait
		for(size_t i1=0; i1<list_A1.size(); ++i1)
		{
			const TA iat0 = list_A0[i0];
			const TA iat1 = list_A1[i1].first;
			const TC &cell1 = list_A1[i1].second;
			const int it0 = ucell.iat2it[iat0];
			const int ia0 = ucell.iat2ia[iat0];
			const int it1 = ucell.iat2it[iat1];
			const int ia1 = ucell.iat2ia[iat1];
			const ModuleBase::Vector3<double> tau0 = ucell.atoms[it0].tau[ia0];
			const ModuleBase::Vector3<double> tau1 = ucell.atoms[it1].tau[ia1];
			const double Rcut
                = std::min(func_cal_Rcut(it0, it1), func_cal_Rcut(it1, it0));
			const Abfs::Vector3_Order<double> R_delta = -tau0+tau1+(RI_Util::array3_to_Vector3(cell1)*ucell.latvec);
			if( R_delta.norm()*ucell.lat0 < Rcut )
			{
				const Tresult Data = func_DPcal_data(it0, it1, R_delta, flags);
				// if(Data.norm(std::numeric_limits<double>::max()) > threshold)
				// {
					#pragma omp critical(LRI_CV_cal_datas)
					Datas[list_A0[i0]][list_A1[i1]] = Data;
				// }
			}
		}
	}
	ModuleBase::timer::tick("LRI_CV", "cal_datas");
	return Datas;
}


template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vs(
	const UnitCell &ucell,
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	const std::map<std::string,bool> &flags)					// + "writable_Vws"
-> std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vs");
	const T_func_DPcal_data<RI::Tensor<Tdata>>
		func_DPcal_V = std::bind(
			&LRI_CV<Tdata>::DPcal_V, this,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	const T_func_cal_Rcut func_cal_Rcut = std::bind(&LRI_CV<Tdata>::cal_V_Rcut,
                                                    this,
                                                    std::placeholders::_1,
                                                    std::placeholders::_2);
	
	return this->cal_datas(ucell,list_A0, list_A1, flags, func_cal_Rcut, func_DPcal_V);
}

template<typename Tdata>
auto LRI_CV<Tdata>::cal_dVs(
	const UnitCell &ucell,
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	const std::map<std::string,bool> &flags)					// + "writable_dVws"
-> std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>>
{
	ModuleBase::TITLE("LRI_CV","cal_dVs");
	const T_func_DPcal_data<std::array<RI::Tensor<Tdata>,3>>
		func_DPcal_dV = std::bind(
			&LRI_CV<Tdata>::DPcal_dV, this,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	
	const T_func_cal_Rcut func_cal_Rcut = std::bind(&LRI_CV<Tdata>::cal_V_Rcut,
                                                    this,
                                                    std::placeholders::_1,
                                                    std::placeholders::_2);

	return this->cal_datas(ucell,list_A0, list_A1, flags, func_cal_Rcut, func_DPcal_dV);
}

template<typename Tdata>
auto LRI_CV<Tdata>::cal_Cs_dCs(
	const UnitCell &ucell,
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	const std::map<std::string,bool> &flags)					// "cal_dC" + "writable_Cws", "writable_dCws", "writable_Vws", "writable_dVws"
-> std::pair<
        std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>,
        std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Cs_dCs");
	const T_func_DPcal_data<std::pair<RI::Tensor<Tdata>, std::array<RI::Tensor<Tdata>,3>>>
		func_DPcal_C_dC = std::bind(
			&LRI_CV<Tdata>::DPcal_C_dC, this,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    const T_func_cal_Rcut func_cal_Rcut = std::bind(&LRI_CV<Tdata>::cal_C_Rcut,
                                                    this,
                                                    std::placeholders::_1,
                                                    std::placeholders::_2);
	
	std::map<TA,std::map<TAC, std::pair<RI::Tensor<Tdata>, std::array<RI::Tensor<Tdata>,3>>>>
		Cs_dCs_tmp = this->cal_datas(ucell,list_A0, list_A1, flags, func_cal_Rcut, func_DPcal_C_dC);

	std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Cs;
    std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>> dCs;
    for (auto& Cs_dCs_A: Cs_dCs_tmp)
        for (auto& Cs_dCs_B: Cs_dCs_A.second) {
            Cs[Cs_dCs_A.first][Cs_dCs_B.first]
                = std::move(std::get<0>(Cs_dCs_B.second));
            if (flags.at("cal_dC"))
                dCs[Cs_dCs_A.first][Cs_dCs_B.first]
                    = std::move(std::get<1>(Cs_dCs_B.second));
        }
	return std::make_pair(Cs, dCs);
}


template<typename Tdata> template<typename To11, typename Tfunc>
To11 LRI_CV<Tdata>::DPcal_o11(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const bool &flag_writable_o11ws,
	pthread_rwlock_t &rwlock_o11,
	std::map<int,std::map<int,std::map<Abfs::Vector3_Order<double>,To11>>> &o11ws,
	const Tfunc &func_cal_o11)
{
	const Abfs::Vector3_Order<double> Rm = -R;
	pthread_rwlock_rdlock(&rwlock_o11);
	const To11 o11_read = RI::Global_Func::find(o11ws, it0, it1, R);
	pthread_rwlock_unlock(&rwlock_o11);

	if(LRI_CV_Tools::exist(o11_read))
	{
		return o11_read;
	}
	else
	{
		pthread_rwlock_rdlock(&rwlock_o11);
		const To11 o11_transform_read = RI::Global_Func::find(o11ws, it1, it0, Rm);
		pthread_rwlock_unlock(&rwlock_o11);

		if(LRI_CV_Tools::exist(o11_transform_read))
		{
			const To11 o11 = LRI_CV_Tools::transform_Rm(o11_transform_read);
			if(flag_writable_o11ws)							// such write may be deleted for memory saving with transform_Rm() every time
			{
				pthread_rwlock_wrlock(&rwlock_o11);
				o11ws[it0][it1][R] = o11;
				pthread_rwlock_unlock(&rwlock_o11);
			}
			return o11;
		}
		else
		{
			const To11 o11 = func_cal_o11(
				it0, it1, ModuleBase::Vector3<double>{0,0,0}, R,
				this->index_abfs, this->index_abfs,
				Matrix_Orbs11::Matrix_Order::AB);
			if(flag_writable_o11ws)
			{
				pthread_rwlock_wrlock(&rwlock_o11);
				o11ws[it0][it1][R] = o11;
				pthread_rwlock_unlock(&rwlock_o11);
			}
			return o11;
		} // end else (!exist(o11_transform_read))
	} // end else (!exist(o11_read))
}

template<typename Tdata>
RI::Tensor<Tdata>
LRI_CV<Tdata>::DPcal_V(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const std::map<std::string,bool> &flags)					// "writable_Vws"
{
	const auto cal_overlap_matrix = std::bind(
		&Matrix_Orbs11::cal_overlap_matrix<Tdata>,
		&this->m_abfs_abfs,
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
	return this->DPcal_o11(it0, it1, R, flags.at("writable_Vws"), this->rwlock_Vw, this->Vws, cal_overlap_matrix);
}

template<typename Tdata>
std::array<RI::Tensor<Tdata>, 3>
LRI_CV<Tdata>::DPcal_dV(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const std::map<std::string,bool> &flags)					// "writable_dVws"
{
	if(ModuleBase::Vector3<double>(0,0,0)==R)
	{
		assert(it0==it1);
		const size_t size = this->index_abfs[it0].count_size;
		const std::array<RI::Tensor<Tdata>, 3> dV = { RI::Tensor<Tdata>({size,size}), RI::Tensor<Tdata>({size,size}), RI::Tensor<Tdata>({size,size}) };
		if(flags.at("writable_dVws"))
		{
			pthread_rwlock_wrlock(&this->rwlock_dVw);
			this->dVws[it0][it1][R] = dV;
			pthread_rwlock_unlock(&this->rwlock_dVw);
		}
		return dV;
	}

	const auto cal_grad_overlap_matrix = std::bind(
		&Matrix_Orbs11::cal_grad_overlap_matrix<Tdata>,
		&this->m_abfs_abfs,
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
	return this->DPcal_o11(it0, it1, R, flags.at("writable_dVws"), this->rwlock_dVw, this->dVws, cal_grad_overlap_matrix);
}


template<typename Tdata>
std::pair<RI::Tensor<Tdata>, std::array<RI::Tensor<Tdata>,3>>
LRI_CV<Tdata>::DPcal_C_dC(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const std::map<std::string,bool> &flags)					// "cal_dC", "writable_Cws", "writable_dCws" + "writable_Vws", "writable_dVws"
{
	using namespace LRI_CV_Tools;

	const Abfs::Vector3_Order<double> Rm = -R;
	pthread_rwlock_rdlock(&this->rwlock_Cw);
	const RI::Tensor<Tdata> C_read = RI::Global_Func::find(this->Cws, it0, it1, R);
	pthread_rwlock_unlock(&this->rwlock_Cw);
	pthread_rwlock_rdlock(&this->rwlock_dCw);
	const std::array<RI::Tensor<Tdata>,3> dC_read = RI::Global_Func::find(this->dCws, it0, it1, R);
	pthread_rwlock_unlock(&this->rwlock_dCw);
	const bool flag_finish_dC = (!flags.at("cal_dC")) || LRI_CV_Tools::exist(dC_read);

	if(!C_read.empty() && flag_finish_dC)
	{
		return std::make_pair(C_read, dC_read);
	}
	else
	{
		if( (ModuleBase::Vector3<double>(0,0,0)==R) && (it0==it1) )
		{
			const RI::Tensor<Tdata>
				A = this->m_abfslcaos_lcaos.template cal_overlap_matrix<Tdata>(
						it0, it1, {0,0,0}, {0,0,0},
						this->index_abfs, this->index_lcaos, this->index_lcaos,
						Matrix_Orbs21::Matrix_Order::A1A2B);
			const RI::Tensor<Tdata> V = this->DPcal_V( it0, it0, {0,0,0}, {{"writable_Vws",true}});
			RI::Tensor<Tdata> L;
			if(GlobalC::exx_info.info_ri.exx_cs_inv_thr > 0) {
				Inverse_Matrix<Tdata> I;
				I.input(V);
				I.cal_inverse(Inverse_Matrix<Tdata>::Method::syev, GlobalC::exx_info.info_ri.exx_cs_inv_thr);
				L = I.output();
			} else {
				L = LRI_CV_Tools::cal_I(V, Inverse_Matrix<Tdata>::Method::potrf, -1.0);
			}

			const RI::Tensor<Tdata> C = RI::Global_Func::convert<Tdata>(0.5) * LRI_CV_Tools::mul1(L,A);					// Attention 0.5!
			if(flags.at("writable_Cws"))
			{
				pthread_rwlock_wrlock(&this->rwlock_Cw);
				this->Cws[it0][it1][{0,0,0}] = C;
				pthread_rwlock_unlock(&this->rwlock_Cw);
			}

			if(flag_finish_dC)
			{
				return std::make_pair(C, dC_read);
			}
			else
			{
				const RI::Shape_Vector sizes = {this->index_abfs[it0].count_size,
				                                this->index_lcaos[it0].count_size,
				                                this->index_lcaos[it0].count_size};
				const std::array<RI::Tensor<Tdata>,3>
					dC({RI::Tensor<Tdata>({sizes}), RI::Tensor<Tdata>({sizes}), RI::Tensor<Tdata>({sizes})});
				if(flags.at("writable_dCws"))
				{
					pthread_rwlock_wrlock(&this->rwlock_dCw);
					this->dCws[it0][it1][{0,0,0}] = dC;
					pthread_rwlock_unlock(&this->rwlock_dCw);
				}
				return std::make_pair(C, dC);
			}
		} // end if( (ModuleBase::Vector3<double>(0,0,0)==R) && (it0==it1) )
		else
		{
			const std::vector<RI::Tensor<Tdata>>
				A = {this->m_abfslcaos_lcaos.template cal_overlap_matrix<Tdata>(
						it0, it1, {0,0,0}, R,
						this->index_abfs, this->index_lcaos, this->index_lcaos,
						Matrix_Orbs21::Matrix_Order::A1A2B),
				     this->m_abfslcaos_lcaos.template cal_overlap_matrix<Tdata>(
						it1, it0, {0,0,0}, Rm,
						this->index_abfs, this->index_lcaos, this->index_lcaos,
						Matrix_Orbs21::Matrix_Order::A1BA2)};

			const std::vector<std::vector<RI::Tensor<Tdata>>>
				V = {{DPcal_V(it0, it0, {0,0,0}, {{"writable_Vws",true}}),
				      DPcal_V(it0, it1, R,       flags)},
				     {DPcal_V(it1, it0, Rm,      flags),
				      DPcal_V(it1, it1, {0,0,0}, {{"writable_Vws",true}})}};

			std::vector<std::vector<RI::Tensor<Tdata>>> L;
			if(GlobalC::exx_info.info_ri.exx_cs_inv_thr > 0) {
				Inverse_Matrix<Tdata> I;
				I.input(V);
				I.cal_inverse(Inverse_Matrix<Tdata>::Method::syev, GlobalC::exx_info.info_ri.exx_cs_inv_thr);
				L = I.output({V[0][0].shape[0], V[1][0].shape[0]},
				             {V[0][0].shape[1], V[0][1].shape[1]});
			} else {
				L = LRI_CV_Tools::cal_I(V, Inverse_Matrix<Tdata>::Method::potrf, -1.0);
			}

			const std::vector<RI::Tensor<Tdata>> C = LRI_CV_Tools::mul2(L,A);
			if(flags.at("writable_Cws"))
			{
				pthread_rwlock_wrlock(&this->rwlock_Cw);
				this->Cws[it0][it1][R] = C[0];
				this->Cws[it1][it0][Rm] = LRI_CV_Tools::transpose12(C[1]);
				pthread_rwlock_unlock(&this->rwlock_Cw);
			}

			if(flag_finish_dC)
			{
				return std::make_pair(C[0], dC_read);
			}
			else
			{
				const std::vector<std::array<RI::Tensor<Tdata>,3>>
					dA = {this->m_abfslcaos_lcaos.template cal_grad_overlap_matrix<Tdata>(
								it0, it1, {0,0,0}, R,
								this->index_abfs, this->index_lcaos, this->index_lcaos,
								Matrix_Orbs21::Matrix_Order::A1A2B),
					      LRI_CV_Tools::negative(
					       this->m_abfslcaos_lcaos.template cal_grad_overlap_matrix<Tdata>(
								it1, it0, {0,0,0}, Rm,
								this->index_abfs, this->index_lcaos, this->index_lcaos,
								Matrix_Orbs21::Matrix_Order::A1BA2))};

				const std::array<RI::Tensor<Tdata>,3> dV_01 = DPcal_dV(it0, it1, R, flags);
				const std::array<RI::Tensor<Tdata>,3> dV_10 = LRI_CV_Tools::negative(DPcal_dV(it1, it0, Rm, flags));

				std::array<std::vector<RI::Tensor<Tdata>>,3>		// dC = L*(dA-dV*C)
					dC_tmp = LRI_CV_Tools::mul2(
							L,
							LRI_CV_Tools::change_order( LRI_CV_Tools::minus(
								dA,
								std::vector<std::array<RI::Tensor<Tdata>,3>>{
									LRI_CV_Tools::mul1(dV_01, C[1]),
									LRI_CV_Tools::mul1(dV_10, C[0])})));
				const std::vector<std::array<RI::Tensor<Tdata>,3>>
					dC = LRI_CV_Tools::change_order(std::move(dC_tmp));
				if(flags.at("writable_dCws"))
				{
					pthread_rwlock_wrlock(&this->rwlock_dCw);
					this->dCws[it0][it1][R] = dC[0];
					this->dCws[it1][it0][Rm] = LRI_CV_Tools::negative(LRI_CV_Tools::transpose12(dC[1]));
					pthread_rwlock_unlock(&this->rwlock_dCw);
				}
				return std::make_pair(C[0], dC[0]);
			} // end else (!flag_finish_dC)
		} // end else ( (ModuleBase::Vector3<double>(0,0,0)!=R) || (it0!=it1) )
	} // end else (!(C_read && flag_finish_dC))
}

// Zc
template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vq1(const K_Vectors* kv,
						const ModulePW::PW_Basis_K* wfc_basis,
						const Structure_Factor& sf,
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						const double &frac)
-> std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vq1");

	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>> datas;
	datas.resize(nks0);

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		const int npw = wfc_basis->npwk[ik];
		std::vector<ModuleBase::ComplexMatrix> abfs_in_G = this->get_orb_q(kv, wfc_basis, sf, this->abfs);
		std::vector<ModuleBase::ComplexMatrix> abfs_ccp_in_G = this->get_orb_q(kv, wfc_basis, sf, this->abfs_ccp);
		std::map<int, int> abfs_nw = Exx_Abfs::Construct_Orbs::get_nw(this->abfs);
		std::map<int, int> abfs_ccp_nw = Exx_Abfs::Construct_Orbs::get_nw(this->abfs_ccp);
		// const int nabfs = Exx_Abfs::Construct_Orbs::get_norb(this->abfs);
		// const int nabfs_ccp = Exx_Abfs::Construct_Orbs::get_norb(this->abfs_ccp);

		//#pragma omp parallel
		for(size_t i0=0; i0!=list_A0.size(); ++i0)
		{
			//#pragma omp for schedule(dynamic) nowait
			for(size_t i1=0; i1!=list_A1.size(); ++i1)
			{
				const TA iat0 = list_A0[i0];
				const TA iat1 = list_A1[i1].first;
				const int it0 = GlobalC::ucell.iat2it[iat0];
				const int ia0 = GlobalC::ucell.iat2ia[iat0];
				const int it1 = GlobalC::ucell.iat2it[iat1];
				const int ia1 = GlobalC::ucell.iat2ia[iat1];
				const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];
				const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];
				int abfs_nw_t = abfs_nw[it0];
				int abfs_ccp_nw_t = abfs_nw[it0];
				RI::Tensor<Tdata> data;//(nabfs, nabfs_ccp);

				for(size_t j0=0; j0!=abfs_nw_t; ++j0)
				{
					const int iw0 = Exx_Abfs::Construct_Orbs::get_itiaiw2iwt(this->abfs, it0, ia0, j0);
					for(size_t j1=0; j1!=abfs_ccp_nw_t; ++j1)
					{
						const int iw1 = Exx_Abfs::Construct_Orbs::get_itiaiw2iwt(this->abfs_ccp, it1, ia1, j1);
						for(size_t ig=0; ig!=npw; ++ig)
						{
							ModuleBase::Vector3<double> gk = wfc_basis->getgcar(ik, ig) - kv->kvec_c[ik];
							std::complex<double> phase = std::exp(ModuleBase::IMAG_UNIT*(gk*(-tau0+tau1)));
							data(iw0, iw1) += std::conj(abfs_in_G[ik](iw0, ig)) * abfs_ccp_in_G[ik](iw1, ig) * phase;
						}
					}
				}

				datas[ik][iat0][iat1] = data * frac;
				//#pragma omp critical(LRI_CV_cal_datas)
			}	
		}
	}

	return datas;
}

// Zd
template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vq2(
	const K_Vectors* kv, 
	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs)
-> std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vq2");

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

	return datas;
}


template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vs_ewald(const K_Vectors* kv, 
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						std::vector<std::map<TA,std::map<TA, RI::Tensor<std::complex<double>>>>>& Vq)
-> std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vs_ewald");

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
				GlobalC::ORB.Phi[it0].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it1].getRcut(),
				GlobalC::ORB.Phi[it1].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it0].getRcut());
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

	return datas;
}
template<typename Tdata>
bool 
LRI_CV<Tdata>::check_Vs(
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs2)
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
				GlobalC::ORB.Phi[it0].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it1].getRcut(),
				GlobalC::ORB.Phi[it1].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it0].getRcut());
			const Abfs::Vector3_Order<double> R_delta = -tau0+tau1+(RI_Util::array3_to_Vector3(cell1)*GlobalC::ucell.latvec);
			if( R_delta.norm()*GlobalC::ucell.lat0 < Rcut )
			{
				std::cout<<"iat0: "<<iat0<<"\tiat1: "<<iat1<<"\tVs1: "<<Vs1[list_A0[i0]][list_A1[i1]](0, 0)<<"\tVs2: "<<Vs2[list_A0[i0]][list_A1[i1]](0, 0)<<std::endl;
			}
		}
	}
	return true;
}

template<typename Tdata>
std::vector<ModuleBase::ComplexMatrix>
LRI_CV<Tdata>::get_orb_q(const K_Vectors* kv, 
						   const ModulePW::PW_Basis_K* wfc_basis, 
						   const Structure_Factor& sf, 
						   const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in
						   )
{
	ModuleBase::TITLE("LRI_CV","get_orb_q");

	// projection to plane wave
	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	int nmax_total = Exx_Abfs::Construct_Orbs::get_nmax_total(orb_in);
	ModuleBase::realArray table_local(orb_in.size(), nmax_total, GlobalV::NQX);
	Wavefunc_in_pw::make_table_q(orb_in, table_local); 
	const int norb = Exx_Abfs::Construct_Orbs::get_norb(orb_in);
	std::vector<ModuleBase::ComplexMatrix> orb_in_Gs;
	orb_in_Gs.resize(nks0);

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		const int npw = wfc_basis->npwk[ik];
		std::vector<ModuleBase::Vector3<double>> gk(npw, ModuleBase::Vector3<double>(0, 0, 0));
		for(size_t ig=0; ig!=npw; ++ig)
			gk[ig] = wfc_basis->getgcar(ik, ig) - kv->kvec_c[ik];
		ModuleBase::ComplexMatrix orb_in_G(norb, npw);
		Wavefunc_in_pw::produce_local_basis_in_pw(ik, gk, orb_in, wfc_basis, sf, orb_in_G, table_local);
		orb_in_Gs[ik] = orb_in_G;
	}

	return orb_in_Gs;
}


// template<typename Tdata>
// std::vector<ModuleBase::Vector3<double>>
// LRI_CV<Tdata>::gen_q_mesh_d(std::vector<double>& nmp_in, std::vector<double>& offset, const int& type)
// {
// 	const int mpnx = nmp_in[0];
//     const int mpny = nmp_in[1];
//     const int mpnz = nmp_in[2];
// 	const int nqs = mpnx * mpny * mpnz;
// 	std::vector<ModuleBase::Vector3<double>> qvec_d(nqs);

// 	for(int x=1; x<=nmp[0]; x++)
// 	{
// 		double v1 = Monkhorst_Pack_formula(type, koffset_in[0], x, mpnx);
// 		for (int y = 1;y <= mpny;y++)
//         {
//             double v2 = Monkhorst_Pack_formula( k_type, koffset_in[1], y, mpny);
// 		    if( std::abs(v2) < 1.0e-10 ) v2 = 0.0;
//             for (int z = 1;z <= mpnz;z++)
//             {
//                 double v3 = Monkhorst_Pack_formula( k_type, koffset_in[2], z, mpnz);
// 				if( std::abs(v3) < 1.0e-10 ) v3 = 0.0;
//                 const int i = mpnx * mpny * (z - 1) + mpnx * (y - 1) + (x - 1);
//                 qvec_d[i].set(v1, v2, v3);
//             }
//         }
//     }

// 	return qvec_d;
// }

// template<typename Tdata>
// std::vector<ModuleBase::Vector3<double>>
// LRI_CV<Tdata>::gen_q_mesh_c(std::vector<double>& nmp_in, std::vector<double>& offset, const int& type)
// {
// 	std::vector<ModuleBase::Vector3<double>> qvec_d = this->gen_q_mesh_d(nmp_in, offset, type);
// 	const int nqs = qvec_d.size();
// 	std::vector<ModuleBase::Vector3<double>> qvec_c(nqs);

// 	for (size_t i = 0;i != nks;++i)
// 		qvec_c[i] = qvec_d[i] * GlobalC::ucell.G;

// 	return qvec_c;
// }

#endif
//=======================
// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#ifndef LRI_CV_HPP
#define LRI_CV_HPP

#include "LRI_CV.h"
#include "LRI_CV_Tools.h"
#include "Inverse_Matrix.h"
#include "module_ri/exx_abfs-abfs-index.h"
#include "exx_abfs-construct_orbs.h"
#include "RI_Util.h"
#include "RI_2D_Comm.h"
#include "module_base/tool_title.h"
#include "module_base/timer.h"
#include <RI/global/Global_Func-1.h>
#include "module_hamilt_lcao/hamilt_lcaodft/wavefunc_in_pw.h"
#include "module_base/realarray.h"
#include <omp.h>
#include<algorithm>

template<typename Tdata>
LRI_CV<Tdata>::LRI_CV()
{
	pthread_rwlock_init(&rwlock_Vw,NULL);
	pthread_rwlock_init(&rwlock_Cw,NULL);
	pthread_rwlock_init(&rwlock_dVw,NULL);
	pthread_rwlock_init(&rwlock_dCw,NULL);
}

template<typename Tdata>
LRI_CV<Tdata>::~LRI_CV()
{
	pthread_rwlock_destroy(&rwlock_Vw);
	pthread_rwlock_destroy(&rwlock_Cw);
	pthread_rwlock_destroy(&rwlock_dVw);
	pthread_rwlock_destroy(&rwlock_dCw);
}


template<typename Tdata>
void LRI_CV<Tdata>::set_orbitals(
	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos_in,
	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs_in,
	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs_ccp_in,
	const double &kmesh_times,
	const double &ccp_rmesh_times_in)
{
	ModuleBase::TITLE("LRI_CV", "set_orbitals");
	ModuleBase::timer::tick("LRI_CV", "set_orbitals");

	this->lcaos = lcaos_in;
	this->abfs = abfs_in;
	this->abfs_ccp = abfs_ccp_in;
	this->ccp_rmesh_times = ccp_rmesh_times_in;

	const ModuleBase::Element_Basis_Index::Range
		range_lcaos = Exx_Abfs::Abfs_Index::construct_range( lcaos );
	this->index_lcaos = ModuleBase::Element_Basis_Index::construct_index( range_lcaos );

	const ModuleBase::Element_Basis_Index::Range
		range_abfs = Exx_Abfs::Abfs_Index::construct_range( abfs );
	this->index_abfs = ModuleBase::Element_Basis_Index::construct_index( range_abfs );

	this->m_abfs_abfs.init( 2, kmesh_times, (1+this->ccp_rmesh_times)/2.0 );
	this->m_abfs_abfs.init_radial( this->abfs_ccp, this->abfs );
	this->m_abfs_abfs.init_radial_table();

	this->m_abfslcaos_lcaos.init( 1, kmesh_times, 1 );
	this->m_abfslcaos_lcaos.init_radial( this->abfs_ccp, this->lcaos, this->lcaos );
	this->m_abfslcaos_lcaos.init_radial_table();

	ModuleBase::timer::tick("LRI_CV", "set_orbitals");
}



template<typename Tdata> template<typename Tresult>
auto LRI_CV<Tdata>::cal_datas(
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	const std::map<std::string,bool> &flags,
	const double &rmesh_times,
	const T_func_DPcal_data<Tresult> &func_DPcal_data)
-> std::map<TA,std::map<TAC,Tresult>>
{
	ModuleBase::TITLE("LRI_CV","cal_datas");
	ModuleBase::timer::tick("LRI_CV", "cal_datas");

	std::map<TA,std::map<TAC,Tresult>> Datas;
	#pragma omp parallel
	for(size_t i0=0; i0<list_A0.size(); ++i0)
	{
		#pragma omp for schedule(dynamic) nowait
		for(size_t i1=0; i1<list_A1.size(); ++i1)
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
				GlobalC::ORB.Phi[it0].getRcut() * rmesh_times + GlobalC::ORB.Phi[it1].getRcut(),
				GlobalC::ORB.Phi[it1].getRcut() * rmesh_times + GlobalC::ORB.Phi[it0].getRcut());
			const Abfs::Vector3_Order<double> R_delta = -tau0+tau1+(RI_Util::array3_to_Vector3(cell1)*GlobalC::ucell.latvec);
			if( R_delta.norm()*GlobalC::ucell.lat0 < Rcut )
			{
				const Tresult Data = func_DPcal_data(it0, it1, R_delta, flags);
//				if(Data.norm(std::numeric_limits<double>::max()) > threshold)
//				{
					#pragma omp critical(LRI_CV_cal_datas)
					Datas[list_A0[i0]][list_A1[i1]] = Data;
//				}
			}
		}
	}
	ModuleBase::timer::tick("LRI_CV", "cal_datas");
	return Datas;
}


template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vs(
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	const std::map<std::string,bool> &flags)					// + "writable_Vws"
-> std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vs");
	const T_func_DPcal_data<RI::Tensor<Tdata>>
		func_DPcal_V = std::bind(
			&LRI_CV<Tdata>::DPcal_V, this,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	return this->cal_datas(list_A0, list_A1, flags, this->ccp_rmesh_times, func_DPcal_V);
}

template<typename Tdata>
auto LRI_CV<Tdata>::cal_dVs(
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	const std::map<std::string,bool> &flags)					// + "writable_dVws"
-> std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>
{
	ModuleBase::TITLE("LRI_CV","cal_dVs");
	const T_func_DPcal_data<std::array<RI::Tensor<Tdata>,3>>
		func_DPcal_dV = std::bind(
			&LRI_CV<Tdata>::DPcal_dV, this,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	return LRI_CV_Tools::change_order(
		this->cal_datas(list_A0, list_A1, flags, this->ccp_rmesh_times, func_DPcal_dV));
}

template<typename Tdata>
auto LRI_CV<Tdata>::cal_Cs_dCs(
	const std::vector<TA> &list_A0,
	const std::vector<TAC> &list_A1,
	const std::map<std::string,bool> &flags)					// "cal_dC" + "writable_Cws", "writable_dCws", "writable_Vws", "writable_dVws"
-> std::pair<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>, std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>>
{
	ModuleBase::TITLE("LRI_CV","cal_Cs_dCs");
	const T_func_DPcal_data<std::pair<RI::Tensor<Tdata>, std::array<RI::Tensor<Tdata>,3>>>
		func_DPcal_C_dC = std::bind(
			&LRI_CV<Tdata>::DPcal_C_dC, this,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	std::map<TA,std::map<TAC, std::pair<RI::Tensor<Tdata>, std::array<RI::Tensor<Tdata>,3>>>>
		Cs_dCs_tmp = this->cal_datas(list_A0, list_A1, flags, std::min(1.0,this->ccp_rmesh_times), func_DPcal_C_dC);

	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Cs;
	std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3> dCs;
	for(auto &Cs_dCs_A : Cs_dCs_tmp)
		for(auto &Cs_dCs_B : Cs_dCs_A.second)
		{
			Cs[Cs_dCs_A.first][Cs_dCs_B.first] = std::move(std::get<0>(Cs_dCs_B.second));
			if(flags.at("cal_dC"))
				for(int ix=0; ix<3; ++ix)
					dCs[ix][Cs_dCs_A.first][Cs_dCs_B.first] = std::move(std::get<1>(Cs_dCs_B.second)[ix]);
		}
	return std::make_pair(Cs, dCs);
}


template<typename Tdata> template<typename To11, typename Tfunc>
To11 LRI_CV<Tdata>::DPcal_o11(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const bool &flag_writable_o11ws,
	pthread_rwlock_t &rwlock_o11,
	std::map<int,std::map<int,std::map<Abfs::Vector3_Order<double>,To11>>> &o11ws,
	const Tfunc &func_cal_o11)
{
	const Abfs::Vector3_Order<double> Rm = -R;
	pthread_rwlock_rdlock(&rwlock_o11);
	const To11 o11_read = RI::Global_Func::find(o11ws, it0, it1, R);
	pthread_rwlock_unlock(&rwlock_o11);

	if(LRI_CV_Tools::exist(o11_read))
	{
		return o11_read;
	}
	else
	{
		pthread_rwlock_rdlock(&rwlock_o11);
		const To11 o11_transform_read = RI::Global_Func::find(o11ws, it1, it0, Rm);
		pthread_rwlock_unlock(&rwlock_o11);

		if(LRI_CV_Tools::exist(o11_transform_read))
		{
			const To11 o11 = LRI_CV_Tools::transform_Rm(o11_transform_read);
			if(flag_writable_o11ws)							// such write may be deleted for memory saving with transform_Rm() every time
			{
				pthread_rwlock_wrlock(&rwlock_o11);
				o11ws[it0][it1][R] = o11;
				pthread_rwlock_unlock(&rwlock_o11);
			}
			return o11;
		}
		else
		{
			const To11 o11 = func_cal_o11(
				it0, it1, ModuleBase::Vector3<double>{0,0,0}, R,
				this->index_abfs, this->index_abfs,
				Matrix_Orbs11::Matrix_Order::AB);
			if(flag_writable_o11ws)
			{
				pthread_rwlock_wrlock(&rwlock_o11);
				o11ws[it0][it1][R] = o11;
				pthread_rwlock_unlock(&rwlock_o11);
			}
			return o11;
		} // end else (!exist(o11_transform_read))
	} // end else (!exist(o11_read))
}

template<typename Tdata>
RI::Tensor<Tdata>
LRI_CV<Tdata>::DPcal_V(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const std::map<std::string,bool> &flags)					// "writable_Vws"
{
	const auto cal_overlap_matrix = std::bind(
		&Matrix_Orbs11::cal_overlap_matrix<Tdata>,
		&this->m_abfs_abfs,
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
	return this->DPcal_o11(it0, it1, R, flags.at("writable_Vws"), this->rwlock_Vw, this->Vws, cal_overlap_matrix);
}

template<typename Tdata>
std::array<RI::Tensor<Tdata>, 3>
LRI_CV<Tdata>::DPcal_dV(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const std::map<std::string,bool> &flags)					// "writable_dVws"
{
	if(ModuleBase::Vector3<double>(0,0,0)==R)
	{
		assert(it0==it1);
		const size_t size = this->index_abfs[it0].count_size;
		const std::array<RI::Tensor<Tdata>, 3> dV = { RI::Tensor<Tdata>({size,size}), RI::Tensor<Tdata>({size,size}), RI::Tensor<Tdata>({size,size}) };
		if(flags.at("writable_dVws"))
		{
			pthread_rwlock_wrlock(&this->rwlock_dVw);
			this->dVws[it0][it1][R] = dV;
			pthread_rwlock_unlock(&this->rwlock_dVw);
		}
		return dV;
	}

	const auto cal_grad_overlap_matrix = std::bind(
		&Matrix_Orbs11::cal_grad_overlap_matrix<Tdata>,
		&this->m_abfs_abfs,
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
	return this->DPcal_o11(it0, it1, R, flags.at("writable_dVws"), this->rwlock_dVw, this->dVws, cal_grad_overlap_matrix);
}


template<typename Tdata>
std::pair<RI::Tensor<Tdata>, std::array<RI::Tensor<Tdata>,3>>
LRI_CV<Tdata>::DPcal_C_dC(
	const int it0,
	const int it1,
	const Abfs::Vector3_Order<double> &R,
	const std::map<std::string,bool> &flags)					// "cal_dC", "writable_Cws", "writable_dCws" + "writable_Vws", "writable_dVws"
{
	using namespace LRI_CV_Tools;

	const Abfs::Vector3_Order<double> Rm = -R;
	pthread_rwlock_rdlock(&this->rwlock_Cw);
	const RI::Tensor<Tdata> C_read = RI::Global_Func::find(this->Cws, it0, it1, R);
	pthread_rwlock_unlock(&this->rwlock_Cw);
	pthread_rwlock_rdlock(&this->rwlock_dCw);
	const std::array<RI::Tensor<Tdata>,3> dC_read = RI::Global_Func::find(this->dCws, it0, it1, R);
	pthread_rwlock_unlock(&this->rwlock_dCw);
	const bool flag_finish_dC = (!flags.at("cal_dC")) || LRI_CV_Tools::exist(dC_read);

	if(!C_read.empty() && flag_finish_dC)
	{
		return std::make_pair(C_read, dC_read);
	}
	else
	{
		if( (ModuleBase::Vector3<double>(0,0,0)==R) && (it0==it1) )
		{
			const RI::Tensor<Tdata>
				A = this->m_abfslcaos_lcaos.template cal_overlap_matrix<Tdata>(
						it0, it1, {0,0,0}, {0,0,0},
						this->index_abfs, this->index_lcaos, this->index_lcaos,
						Matrix_Orbs21::Matrix_Order::A1A2B);
			const RI::Tensor<Tdata> V = this->DPcal_V( it0, it0, {0,0,0}, {{"writable_Vws",true}});
			RI::Tensor<Tdata> L;
			if(GlobalC::exx_info.info_ri.exx_cs_inv_thr > 0) {
				Inverse_Matrix<Tdata> I;
				I.input(V);
				I.cal_inverse(Inverse_Matrix<Tdata>::Method::syev, GlobalC::exx_info.info_ri.exx_cs_inv_thr);
				L = I.output();
			} else {
				L = LRI_CV_Tools::cal_I(V, Inverse_Matrix<Tdata>::Method::potrf, -1.0);
			}

			const RI::Tensor<Tdata> C = RI::Global_Func::convert<Tdata>(0.5) * LRI_CV_Tools::mul1(L,A);					// Attention 0.5!
			if(flags.at("writable_Cws"))
			{
				pthread_rwlock_wrlock(&this->rwlock_Cw);
				this->Cws[it0][it1][{0,0,0}] = C;
				pthread_rwlock_unlock(&this->rwlock_Cw);
			}

			if(flag_finish_dC)
			{
				return std::make_pair(C, dC_read);
			}
			else
			{
				const RI::Shape_Vector sizes = {this->index_abfs[it0].count_size,
				                                this->index_lcaos[it0].count_size,
				                                this->index_lcaos[it0].count_size};
				const std::array<RI::Tensor<Tdata>,3>
					dC({RI::Tensor<Tdata>({sizes}), RI::Tensor<Tdata>({sizes}), RI::Tensor<Tdata>({sizes})});
				if(flags.at("writable_dCws"))
				{
					pthread_rwlock_wrlock(&this->rwlock_dCw);
					this->dCws[it0][it1][{0,0,0}] = dC;
					pthread_rwlock_unlock(&this->rwlock_dCw);
				}
				return std::make_pair(C, dC);
			}
		} // end if( (ModuleBase::Vector3<double>(0,0,0)==R) && (it0==it1) )
		else
		{
			const std::vector<RI::Tensor<Tdata>>
				A = {this->m_abfslcaos_lcaos.template cal_overlap_matrix<Tdata>(
						it0, it1, {0,0,0}, R,
						this->index_abfs, this->index_lcaos, this->index_lcaos,
						Matrix_Orbs21::Matrix_Order::A1A2B),
				     this->m_abfslcaos_lcaos.template cal_overlap_matrix<Tdata>(
						it1, it0, {0,0,0}, Rm,
						this->index_abfs, this->index_lcaos, this->index_lcaos,
						Matrix_Orbs21::Matrix_Order::A1BA2)};

			const std::vector<std::vector<RI::Tensor<Tdata>>>
				V = {{DPcal_V(it0, it0, {0,0,0}, {{"writable_Vws",true}}),
				      DPcal_V(it0, it1, R,       flags)},
				     {DPcal_V(it1, it0, Rm,      flags),
				      DPcal_V(it1, it1, {0,0,0}, {{"writable_Vws",true}})}};

			std::vector<std::vector<RI::Tensor<Tdata>>> L;
			if(GlobalC::exx_info.info_ri.exx_cs_inv_thr > 0) {
				Inverse_Matrix<Tdata> I;
				I.input(V);
				I.cal_inverse(Inverse_Matrix<Tdata>::Method::syev, GlobalC::exx_info.info_ri.exx_cs_inv_thr);
				L = I.output({V[0][0].shape[0], V[1][0].shape[0]},
				             {V[0][0].shape[1], V[0][1].shape[1]});
			} else {
				L = LRI_CV_Tools::cal_I(V, Inverse_Matrix<Tdata>::Method::potrf, -1.0);
			}

			const std::vector<RI::Tensor<Tdata>> C = LRI_CV_Tools::mul2(L,A);
			if(flags.at("writable_Cws"))
			{
				pthread_rwlock_wrlock(&this->rwlock_Cw);
				this->Cws[it0][it1][R] = C[0];
				this->Cws[it1][it0][Rm] = LRI_CV_Tools::transpose12(C[1]);
				pthread_rwlock_unlock(&this->rwlock_Cw);
			}

			if(flag_finish_dC)
			{
				return std::make_pair(C[0], dC_read);
			}
			else
			{
				const std::vector<std::array<RI::Tensor<Tdata>,3>>
					dA = {this->m_abfslcaos_lcaos.template cal_grad_overlap_matrix<Tdata>(
								it0, it1, {0,0,0}, R,
								this->index_abfs, this->index_lcaos, this->index_lcaos,
								Matrix_Orbs21::Matrix_Order::A1A2B),
					      LRI_CV_Tools::negative(
					       this->m_abfslcaos_lcaos.template cal_grad_overlap_matrix<Tdata>(
								it1, it0, {0,0,0}, Rm,
								this->index_abfs, this->index_lcaos, this->index_lcaos,
								Matrix_Orbs21::Matrix_Order::A1BA2))};

				const std::array<RI::Tensor<Tdata>,3> dV_01 = DPcal_dV(it0, it1, R, flags);
				const std::array<RI::Tensor<Tdata>,3> dV_10 = LRI_CV_Tools::negative(DPcal_dV(it1, it0, Rm, flags));

				std::array<std::vector<RI::Tensor<Tdata>>,3>		// dC = L*(dA-dV*C)
					dC_tmp = LRI_CV_Tools::mul2(
							L,
							LRI_CV_Tools::change_order( LRI_CV_Tools::minus(
								dA,
								std::vector<std::array<RI::Tensor<Tdata>,3>>{
									LRI_CV_Tools::mul1(dV_01, C[1]),
									LRI_CV_Tools::mul1(dV_10, C[0])})));
				const std::vector<std::array<RI::Tensor<Tdata>,3>>
					dC = LRI_CV_Tools::change_order(std::move(dC_tmp));
				if(flags.at("writable_dCws"))
				{
					pthread_rwlock_wrlock(&this->rwlock_dCw);
					this->dCws[it0][it1][R] = dC[0];
					this->dCws[it1][it0][Rm] = LRI_CV_Tools::negative(LRI_CV_Tools::transpose12(dC[1]));
					pthread_rwlock_unlock(&this->rwlock_dCw);
				}
				return std::make_pair(C[0], dC[0]);
			} // end else (!flag_finish_dC)
		} // end else ( (ModuleBase::Vector3<double>(0,0,0)!=R) || (it0!=it1) )
	} // end else (!(C_read && flag_finish_dC))
}

// Zc
template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vq1(const K_Vectors* kv,
						const ModulePW::PW_Basis_K* wfc_basis,
						const Structure_Factor& sf,
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1)
-> std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vq1");

	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>> datas;
	datas.resize(nks0);
	const size_t nabfs = Exx_Abfs::Construct_Orbs::get_norb(this->abfs);
	const size_t nabfs_ccp = Exx_Abfs::Construct_Orbs::get_norb(this->abfs_ccp);
	std::vector<ModuleBase::ComplexMatrix> abfs_in_G = this->get_orb_q(kv, wfc_basis, sf, this->abfs);
	std::vector<ModuleBase::ComplexMatrix> abfs_ccp_in_G = this->get_orb_q(kv, wfc_basis, sf, this->abfs_ccp);
	std::map<int, int> abfs_nw = Exx_Abfs::Construct_Orbs::get_nw(this->abfs);
	std::map<int, int> abfs_ccp_nw = Exx_Abfs::Construct_Orbs::get_nw(this->abfs_ccp);

	// for (int i = 0; i < GlobalC::ucell.nat; i++)
	// {
	// 	int a = GlobalC::ucell.iat2ia[i];
    //     int t = GlobalC::ucell.iat2it[i];
    //     for(int j = 0; j < GlobalC::ucell.atoms[t].nw; ++j)
	// 	{
	// 		std::cout<<"norb1:\t"<<GlobalC::ucell.itiaiw2iwt(t, a, j)<<"norb2:\t"<<Exx_Abfs::Construct_Orbs::get_itiaiw2iwt(this->lcaos, t, a, j)<<std::endl;
	// 	}
	// }

	std::set<TA> unique_set_A1;
	for(const auto& pair : list_A1)
		unique_set_A1.insert(pair.first);
	std::vector<TA> unique_list_A1(unique_set_A1.begin(), unique_set_A1.end());

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		const int npw = wfc_basis->npwk[ik];

		//#pragma omp parallel
		for(size_t i0=0; i0!=list_A0.size(); ++i0)
		{
			//#pragma omp for schedule(dynamic) nowait
			for(size_t i1=0; i1!=unique_list_A1.size(); ++i1)
			{
				const TA iat0 = list_A0[i0];
				const TA iat1 = unique_list_A1[i1];
				const int it0 = GlobalC::ucell.iat2it[iat0];
				const int ia0 = GlobalC::ucell.iat2ia[iat0];
				const int it1 = GlobalC::ucell.iat2it[iat1];
				const int ia1 = GlobalC::ucell.iat2ia[iat1];
				const ModuleBase::Vector3<double> tau0 = GlobalC::ucell.atoms[it0].tau[ia0];
				const ModuleBase::Vector3<double> tau1 = GlobalC::ucell.atoms[it1].tau[ia1];
				int abfs_nw_t = abfs_nw[it0];
				int abfs_ccp_nw_t = abfs_nw[it0];
				RI::Tensor<std::complex<double>> data({nabfs, nabfs_ccp});

				for(size_t j0=0; j0!=abfs_nw_t; ++j0)
				{
					const int iw0 = Exx_Abfs::Construct_Orbs::get_itiaiw2iwt(this->abfs, it0, ia0, j0);
					for(size_t j1=0; j1!=abfs_ccp_nw_t; ++j1)
					{
						const int iw1 = Exx_Abfs::Construct_Orbs::get_itiaiw2iwt(this->abfs_ccp, it1, ia1, j1);
						for(size_t ig=0; ig!=npw; ++ig)
						{
							ModuleBase::Vector3<double> gk = wfc_basis->getgcar(ik, ig) - kv->kvec_c[ik];
							std::complex<double> phase = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT*(gk*(-tau0+tau1)));
							data(iw0, iw1) += std::conj(abfs_in_G[ik](iw0, ig)) * abfs_ccp_in_G[ik](iw1, ig) * phase;
						}
					}
				}
				//#pragma omp critical(LRI_CV_cal_Vq1)
				datas[ik][iat0][iat1] = data;
			}	
		}
	}

	return datas;
}

// Zd
template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vq2(
	const K_Vectors* kv, 
	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs)
-> std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vq2");

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

	return datas;
}


template<typename Tdata>
auto LRI_CV<Tdata>::cal_Vs_ewald(const K_Vectors* kv, 
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						std::vector<std::map<TA,std::map<TA, RI::Tensor<std::complex<double>>>>>& Vq)
-> std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
{
	ModuleBase::TITLE("LRI_CV","cal_Vs_ewald");

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
				GlobalC::ORB.Phi[it0].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it1].getRcut(),
				GlobalC::ORB.Phi[it1].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it0].getRcut());
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

	return datas;
}


template<typename Tdata>
bool 
LRI_CV<Tdata>::check_Vs(
						const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs2)
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
				GlobalC::ORB.Phi[it0].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it1].getRcut(),
				GlobalC::ORB.Phi[it1].getRcut() * this->ccp_rmesh_times + GlobalC::ORB.Phi[it0].getRcut());
			const Abfs::Vector3_Order<double> R_delta = -tau0+tau1+(RI_Util::array3_to_Vector3(cell1)*GlobalC::ucell.latvec);
			if( R_delta.norm()*GlobalC::ucell.lat0 < Rcut )
			{
				std::cout<<"iat0: "<<iat0<<"\tiat1: "<<iat1<<"\tVs1: "<<Vs1[list_A0[i0]][list_A1[i1]](0, 0)<<"\tVs2: "<<Vs2[list_A0[i0]][list_A1[i1]](0, 0)<<std::endl;
			}
		}
	}
	return true;
}

template<typename Tdata>
std::vector<ModuleBase::ComplexMatrix>
LRI_CV<Tdata>::get_orb_q(const K_Vectors* kv, 
						   const ModulePW::PW_Basis_K* wfc_basis, 
						   const Structure_Factor& sf, 
						   const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in
						   )
{
	ModuleBase::TITLE("LRI_CV","get_orb_q");

	// projection to plane wave
	const int nspin0 = std::map<int,int>{{1,1}, {2,2}, {4,1}}.at(GlobalV::NSPIN);
	const int nks0 = kv->nks/nspin0;
	int nmax_total = Exx_Abfs::Construct_Orbs::get_nmax_total(orb_in);
	ModuleBase::realArray table_local(orb_in.size(), nmax_total, GlobalV::NQX);
	Wavefunc_in_pw::make_table_q(orb_in, table_local); 
	const int norb = Exx_Abfs::Construct_Orbs::get_norb(orb_in);
	std::vector<ModuleBase::ComplexMatrix> orb_in_Gs;
	orb_in_Gs.resize(nks0);

	for(size_t ik=0; ik!=nks0; ++ik)
	{
		const int npw = wfc_basis->npwk[ik];
		std::vector<ModuleBase::Vector3<double>> gk(npw, ModuleBase::Vector3<double>(0, 0, 0));
		for(size_t ig=0; ig!=npw; ++ig)
			gk[ig] = wfc_basis->getgcar(ik, ig) - kv->kvec_c[ik];
		ModuleBase::ComplexMatrix orb_in_G(norb, npw);
		Wavefunc_in_pw::produce_local_basis_in_pw(ik, gk, orb_in, wfc_basis, sf, orb_in_G, table_local);
		orb_in_Gs[ik] = orb_in_G;
	}

	return orb_in_Gs;
}


// template<typename Tdata>
// std::vector<ModuleBase::Vector3<double>>
// LRI_CV<Tdata>::gen_q_mesh_d(std::vector<double>& nmp_in, std::vector<double>& offset, const int& type)
// {
// 	const int mpnx = nmp_in[0];
//     const int mpny = nmp_in[1];
//     const int mpnz = nmp_in[2];
// 	const int nqs = mpnx * mpny * mpnz;
// 	std::vector<ModuleBase::Vector3<double>> qvec_d(nqs);

// 	for(int x=1; x<=nmp[0]; x++)
// 	{
// 		double v1 = Monkhorst_Pack_formula(type, koffset_in[0], x, mpnx);
// 		for (int y = 1;y <= mpny;y++)
//         {
//             double v2 = Monkhorst_Pack_formula( k_type, koffset_in[1], y, mpny);
// 		    if( std::abs(v2) < 1.0e-10 ) v2 = 0.0;
//             for (int z = 1;z <= mpnz;z++)
//             {
//                 double v3 = Monkhorst_Pack_formula( k_type, koffset_in[2], z, mpnz);
// 				if( std::abs(v3) < 1.0e-10 ) v3 = 0.0;
//                 const int i = mpnx * mpny * (z - 1) + mpnx * (y - 1) + (x - 1);
//                 qvec_d[i].set(v1, v2, v3);
//             }
//         }
//     }

// 	return qvec_d;
// }

// template<typename Tdata>
// std::vector<ModuleBase::Vector3<double>>
// LRI_CV<Tdata>::gen_q_mesh_c(std::vector<double>& nmp_in, std::vector<double>& offset, const int& type)
// {
// 	std::vector<ModuleBase::Vector3<double>> qvec_d = this->gen_q_mesh_d(nmp_in, offset, type);
// 	const int nqs = qvec_d.size();
// 	std::vector<ModuleBase::Vector3<double>> qvec_c(nqs);

// 	for (size_t i = 0;i != nks;++i)
// 		qvec_c[i] = qvec_d[i] * GlobalC::ucell.G;

// 	return qvec_c;
// }

#endif
