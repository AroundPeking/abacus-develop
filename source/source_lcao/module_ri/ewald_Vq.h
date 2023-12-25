//=======================
// AUTHOR : jiyy
// DATE :   2023-12-08
//=======================

#ifndef EWALD_VQ_H
#define EWALD_VQ_H

#include "module_cell/klist.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_hamilt_pw/hamilt_pwdft/structure_factor.h"
#include "module_base/complexmatrix.h"
#include "module_base/abfs-vector3_order.h"

#include <RI/global/Tensor.h>

#include <vector>
#include <map>

template<typename Tdata>
class Ewald_Vq
{
private:
	using TA = int;
	using TC = std::array<int,3>;
	using TAC = std::pair<TA,TC>;

public:

    std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
	cal_Vs_ewald(const K_Vectors* kv, 
			const std::vector<TA> &list_A0,
			const std::vector<TAC> &list_A1,
			std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>& Vq,
            const double &ccp_rmesh_times);

	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
	cal_Vq1(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
            const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_ccp,
            const K_Vectors* kv,
			const ModulePW::PW_Basis_K* wfc_basis,
			const std::vector<TA> &list_A0,
			const std::vector<TAC> &list_A1,
			const double& gk_ecut
			);

	std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>
	cal_Vq2(const K_Vectors* kv, 
			std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs);

	bool check_Vq(const K_Vectors* kv, 
				const std::vector<TA> &list_A0,
				const std::vector<TAC> &list_A1,
				std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>& Vq1,
				std::vector<std::map<TA,std::map<TA,RI::Tensor<std::complex<double>>>>>& Vq2);

	bool check_Vs(const std::vector<TA> &list_A0,
						const std::vector<TAC> &list_A1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs1,
						std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>& Vs2,
                        const double &ccp_rmesh_times);

private:

    std::pair<std::vector<std::vector<ModuleBase::Vector3<double>>>, std::vector<std::vector<ModuleBase::ComplexMatrix>>>
	get_orb_q(const K_Vectors* kv, 
			const ModulePW::PW_Basis_K* wfc_basis,
			const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in,
			const double& gk_ecut);
	std::vector<ModuleBase::ComplexMatrix>
	produce_local_basis_in_pw(const int& ik,
							std::vector<ModuleBase::Vector3<double>>& gk,
							const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in,
                            const K_Vectors* kv,
							const ModulePW::PW_Basis_K* wfc_basis,
                            const ModuleBase::realArray& table_local,
							const double& gk_ecut);
	std::vector<int> get_npwk(const K_Vectors* kv, const ModulePW::PW_Basis_K* wfc_basis, const double& gk_ecut);
	ModuleBase::Vector3<double> get_gcar(const ModulePW::PW_Basis_K* wfc_basis, const int ig);
};

#include "ewald_Vq.hpp"

#endif