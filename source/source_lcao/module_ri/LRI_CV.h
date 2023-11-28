//=======================
// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#ifndef LRI_CV_H
#define LRI_CV_H

#include "Matrix_Orbs11.h"
#include "Matrix_Orbs21.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_base/abfs-vector3_order.h"
#include "module_base/element_basis_index.h"
#include "module_base/complexmatrix.h"
#include "module_cell/klist.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_hamilt_pw/hamilt_pwdert/structure_factor.h"

#include <RI/global/Tensor.h>
#include <RI/global/Global_Func-2.h>

#include <vector>
#include <map>
#include <functional>
#include <pthread.h>

template<typename Tdata>
class LRI_CV
{
private:
	using TA = int;
	using TC = std::array<int,3>;
	using TAC = std::pair<TA,TC>;
	using Tdata_real = RI::Global_Func::To_Real_t<Tdata>;

public:
	LRI_CV();
	~LRI_CV();

	void set_orbitals(
		const UnitCell &ucell,
        const LCAO_Orbitals& orb,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos_in,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs_in,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs_ccp_in,
		const double &kmesh_times,
		const double &ccp_rmesh_times_in);
	inline std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
	cal_Vs(
		const UnitCell &ucell,
		const std::vector<TA> &list_A0,
		const std::vector<TAC> &list_A1,
		const std::map<std::string,bool> &flags);						// "writable_Vws"
	inline std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>>
	cal_dVs(
		const UnitCell &ucell,
		const std::vector<TA> &list_A0,
		const std::vector<TAC> &list_A1,
		const std::map<std::string,bool> &flags);						// "writable_dVws"
	std::pair<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>,
              std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>>>
	cal_Cs_dCs(
		const UnitCell &ucell,
		const std::vector<TA> &list_A0,
		const std::vector<TAC> &list_A1,
		const std::map<std::string,bool> &flags);						// "cal_dC", "writable_Cws", "writable_dCws", "writable_Vws", "writable_dVws"
	
	size_t get_index_abfs_size(const size_t &iat){return this->index_abfs[iat].count_size; }

	
	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
	cal_Vs_ewald(const K_Vectors* kv, 
			const std::vector<TA> &list_A0,
			const std::vector<TAC> &list_A1,
			std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>& Vq);

	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
	cal_Vq1(const K_Vectors* kv,
			const ModulePW::PW_Basis_K* wfc_basis,
			const Structure_Factor& sf,
			const std::vector<TA> &list_A0,
			const std::vector<TAC> &list_A1,
			const double &frac);

	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
	cal_Vq2(const K_Vectors* kv, 
			std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs);

	// bool check_Vs(const std::vector<TA> &list_A0,
	// 					const std::vector<TAC> &list_A1,
	// 					std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs1,
	// 					std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs2);

	// std::vector<ModuleBase::Vector3<double>>
	// gen_q_mesh_d(std::vector<double>& nmp_in, std::vector<double>& offset, const int& type); // gen q mesh in direct

	// std::vector<ModuleBase::Vector3<double>>
	// gen_q_mesh_c(std::vector<double>& nmp_in, std::vector<double>& offset, const int& type); // gen q mesh in cartesian

private:
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> lcaos;
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs;
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_ccp;
	ModuleBase::Element_Basis_Index::IndexLNM index_lcaos;
	ModuleBase::Element_Basis_Index::IndexLNM index_abfs;
	std::vector<double> lcaos_rcut;
    std::vector<double> abfs_ccp_rcut;

public:
	std::map<int,std::map<int,std::map<Abfs::Vector3_Order<double>,RI::Tensor<Tdata>>>> Vws;
	std::map<int,std::map<int,std::map<Abfs::Vector3_Order<double>,RI::Tensor<Tdata>>>> Cws;
	std::map<int,std::map<int,std::map<Abfs::Vector3_Order<double>,std::array<RI::Tensor<Tdata>,3>>>> dVws;
	std::map<int,std::map<int,std::map<Abfs::Vector3_Order<double>,std::array<RI::Tensor<Tdata>,3>>>> dCws;
private:
	pthread_rwlock_t rwlock_Vw;
	pthread_rwlock_t rwlock_Cw;
	pthread_rwlock_t rwlock_dVw;
	pthread_rwlock_t rwlock_dCw;

	Matrix_Orbs11 m_abfs_abfs;
	Matrix_Orbs21 m_abfslcaos_lcaos;

	template<typename Tresult>
	using T_func_DPcal_data = std::function<Tresult(
		const int it0,
		const int it1,
		const Abfs::Vector3_Order<double> &R,
		const std::map<std::string,bool> &flags)>;
	using T_func_cal_Rcut = std::function<double(const int it0, const int it1)>;
	template<typename Tresult>
	std::map<TA,std::map<TAC,Tresult>>
	cal_datas(
		const UnitCell &ucell,
        const std::vector<TA>& list_A0,
        const std::vector<TAC>& list_A1,
        const std::map<std::string, bool>& flags,
        const T_func_cal_Rcut& func_cal_Rcut,
        const T_func_DPcal_data<Tresult>& func_DPcal_data);

    inline double cal_V_Rcut(const int it0, const int it1);
    inline double cal_C_Rcut(const int it0, const int it1);

	inline RI::Tensor<Tdata>
	DPcal_V(
		const int it0,
		const int it1,
		const Abfs::Vector3_Order<double> &R,
		const std::map<std::string,bool> &flags);						// "writable_Vws"
	inline std::array<RI::Tensor<Tdata>,3>
	DPcal_dV(
		const int it0,
		const int it1,
		const Abfs::Vector3_Order<double> &R,
		const std::map<std::string,bool> &flags);						// "writable_dVws"
	std::pair<RI::Tensor<Tdata>, std::array<RI::Tensor<Tdata>,3>>
	DPcal_C_dC(
		const int it0,
		const int it1,
		const Abfs::Vector3_Order<double> &R,
		const std::map<std::string,bool> &flags);						// "cal_dC", "writable_Cws", "writable_dCws", "writable_Vws", "writable_dVws"

	template<typename To11, typename Tfunc>
	To11 DPcal_o11(
		const int it0,
		const int it1,
		const Abfs::Vector3_Order<double> &R,
		const bool &flag_writable_o11ws,
		pthread_rwlock_t &rwlock_o11,
		std::map<int,std::map<int,std::map<Abfs::Vector3_Order<double>,To11>>> &o11ws,
		const Tfunc &func_cal_o11);

	std::vector<ModuleBase::ComplexMatrix>
	get_orb_q(const K_Vectors* kv, 
			const ModulePW::PW_Basis_K* wfc_basis, 
			const Structure_Factor& sf, 
			const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in);

};

#include "LRI_CV.hpp"

#endif
