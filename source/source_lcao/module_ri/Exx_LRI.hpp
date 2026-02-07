//=======================
// AUTHOR : Peize Lin
#include "source_io/module_parameter/parameter.h"
// DATE :   2022-08-17
//=======================

#ifndef EXX_LRI_HPP
#define EXX_LRI_HPP

#include "Exx_LRI.h"
#include "RI_2D_Comm.h"
#include "RI_Util.h"
#include "source_lcao/module_ri/exx_abfs-construct_orbs.h"
#include "source_lcao/module_ri/exx_abfs-io.h"
#include "source_lcao/module_ri/conv_coulomb_pot_k.h"
#include "source_base/tool_title.h"
#include "source_base/timer.h"
#include "source_lcao/module_ri/serialization_cereal.h"
#include "source_lcao/module_ri/Mix_DMk_2D.h"
#include "source_basis/module_ao/parallel_orbitals.h"

#include <RI/distribute/Distribute_Equally.h>
#include <RI/global/Map_Operator-3.h>

#include <fstream>
#include <string>

template<typename Tdata>
void Exx_LRI<Tdata>::init(const MPI_Comm &mpi_comm_in,
						  const UnitCell &ucell,
						  const K_Vectors &kv_in,
						  const LCAO_Orbitals& orb)
{
	ModuleBase::TITLE("Exx_LRI","init");
	ModuleBase::timer::tick("Exx_LRI", "init");

	this->mpi_comm = mpi_comm_in;
	this->p_kv = &kv_in;
	this->orb_cutoff_ = orb.cutoffs();

	this->lcaos = Exx_Abfs::Construct_Orbs::change_orbs( orb, this->info.kmesh_times );

	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>
		abfs_same_atom = Exx_Abfs::Construct_Orbs::abfs_same_atom(ucell, orb, this->lcaos, this->info.kmesh_times, this->info.pca_threshold );
	if(this->info.files_abfs.empty())
		{ this->abfs = abfs_same_atom;}
	else
		{ this->abfs = Exx_Abfs::IO::construct_abfs( abfs_same_atom, orb, this->info.files_abfs, this->info.kmesh_times ); 	}
	Exx_Abfs::Construct_Orbs::print_orbs_size(ucell, this->abfs, GlobalV::ofs_running);

	// Save original abfs before rotation for later Gaussian fitting
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_original = this->abfs;

	if (this->info.coul_moment == true)
	{
		if (!moment_abfs)
			moment_abfs = new Moment_abfs<Tdata>(GlobalC::exx_info.info_ri);

		ModuleBase::TITLE("cal_multipole_start");
		moment_abfs->cal_multipole(this->abfs);
		ModuleBase::TITLE("cal_multipole_end");

		if (this->info.rotate_abfs == true)
		{
			ModuleBase::TITLE("rotate_abfs_start");
			moment_abfs->rotate_abfs(this->abfs);
			ModuleBase::TITLE("rotate_abfs_end");
		}
	}

	// Calculate modified rmesh_times for coul_moment case
	double ccp_rmesh_times_modified = this->info.ccp_rmesh_times;
	if (this->info.coul_moment == true)
	{
		// ccp_rcut = abfs_rcut + rcut_max to calculate VR in the range of Rcut
		// rcut calculated in k-space
		// To keep safe for different abfs_rcut, use rcut_max*2 / rcut_min
		// For Cs = (ij|v)(v|u)^{-1}, 2*rcut_max is enough due to ij must has overlap.
		double rcut_max = 0.0;
		double rcut_min = 0.0;
		for (const double& rc: this->orb_cutoff_)
		{
			rcut_max = std::max(rcut_max, rc);
			rcut_min = (rcut_min == 0.0) ? rc : std::min(rcut_min, rc);
		}
		ccp_rmesh_times_modified = (rcut_max * 2.0) / rcut_min;
		std::cout << "ccp_rmesh_times is modified to " << ccp_rmesh_times_modified << " due to moment method." << std::endl;
	}

	for( size_t T=0; T!=this->abfs.size(); ++T )
		{ GlobalC::exx_info.info_ri.abfs_Lmax = std::max( GlobalC::exx_info.info_ri.abfs_Lmax, static_cast<int>(this->abfs[T].size())-1 ); }

	this->coulomb_settings = RI_Util::update_coulomb_settings(this->info.coulomb_param, ucell, this->p_kv);

	bool init_MGT = true;
	for(const auto &settings_list : this->coulomb_settings)
	{
		this->exx_objs[settings_list.first].abfs_ccp = Conv_Coulomb_Pot_K::cal_orbs_ccp(this->abfs, settings_list.second.second, ccp_rmesh_times_modified);
		this->exx_objs[settings_list.first].cv.set_orbitals(ucell, orb,
															this->lcaos, this->abfs, this->exx_objs[settings_list.first].abfs_ccp,
															this->info.kmesh_times, this->MGT, init_MGT, settings_list.second.first );
		init_MGT = false; // only init once
		if (settings_list.first == Conv_Coulomb_Pot_K::Coulomb_Method::Ewald)
		{
			const auto &coulomb_param = settings_list.second.second;
			if (this->info.rotate_abfs == true)
			{
				// Fit Gaussian to original abfs and rotate with same coefficients
				ModuleBase::TITLE("fit_gaussian_before_rotate_start");
				std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs;
				g_abfs = this->exx_objs[settings_list.first].evq.init_gauss(abfs_original);
				ModuleBase::TITLE("fit_gaussian_before_rotate_end");

				ModuleBase::TITLE("rotate_gaussian_start");
				// moment_abfs->rotate_abfs(g_abfs);
				ModuleBase::TITLE("rotate_gaussian_end");

				// Pass both abfs (rotated non-Gaussian) and g_abfs (Gaussian-fitted and rotated)
				this->exx_objs[settings_list.first].evq.init(ucell, orb,
															 this->mpi_comm, this->p_kv, this->lcaos, this->abfs, g_abfs,
															 coulomb_param, this->MGT, this->info.ccp_rmesh_times, this->info.kmesh_times);
			}
			else
			{
				this->exx_objs[settings_list.first].evq.init(ucell, orb,
															 this->mpi_comm, this->p_kv, this->lcaos, this->abfs,
															 coulomb_param, this->MGT, this->info.ccp_rmesh_times, this->info.kmesh_times);
			}
		}
	}

	ModuleBase::timer::tick("Exx_LRI", "init");
}

template<typename Tdata>
void Exx_LRI<Tdata>::init(const MPI_Comm& mpi_comm_in,
                          const UnitCell& ucell,
                          const K_Vectors& kv_in,
                          const LCAO_Orbitals& orb,
                          const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_s)
{
	ModuleBase::TITLE("Exx_LRI","init");
	ModuleBase::timer::tick("Exx_LRI", "init");

	this->mpi_comm = mpi_comm_in;
	this->p_kv = &kv_in;
	this->orb_cutoff_ = orb.cutoffs();

	this->lcaos = Exx_Abfs::Construct_Orbs::change_orbs( orb, this->info.kmesh_times );
	this->abfs = abfs_s;
	// Save original abfs before rotation for later Gaussian fitting
	std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_original = this->abfs;

	if (this->info.coul_moment == true)
	{
		if (!moment_abfs)
			moment_abfs = new Moment_abfs<Tdata>(GlobalC::exx_info.info_ri);

		ModuleBase::TITLE("cal_multipole_start");
		moment_abfs->cal_multipole(this->abfs);
		ModuleBase::TITLE("cal_multipole_end");

		if (this->info.rotate_abfs == true)
		{
			ModuleBase::TITLE("rotate_abfs_start");
			moment_abfs->rotate_abfs(this->abfs);
			ModuleBase::TITLE("rotate_abfs_end");
		}
	}

	// Calculate modified rmesh_times for coul_moment case
	double ccp_rmesh_times_modified = this->info.ccp_rmesh_times;
	if (this->info.coul_moment == true)
	{
		// ccp_rcut = abfs_rcut + rcut_max to calculate VR in the range of Rcut
		// rcut calculated in k-space
		// To keep safe for different abfs_rcut, use rcut_max*2 / rcut_min
		// For Cs = (ij|v)(v|u)^{-1}, 2*rcut_max is enough due to ij must has overlap.
		double rcut_max = 0.0;
		double rcut_min = 0.0;
		for (const double& rc: this->orb_cutoff_)
		{
			rcut_max = std::max(rcut_max, rc);
			rcut_min = (rcut_min == 0.0) ? rc : std::min(rcut_min, rc);
		}
		ccp_rmesh_times_modified = (rcut_max * 2.0) / rcut_min;
		std::cout << "ccp_rmesh_times is modified to " << ccp_rmesh_times_modified << " due to moment method." << std::endl;
	}

	for( size_t T=0; T!=this->abfs.size(); ++T )
		{ GlobalC::exx_info.info_ri.abfs_Lmax = std::max( GlobalC::exx_info.info_ri.abfs_Lmax, static_cast<int>(this->abfs[T].size())-1 ); }

	this->coulomb_settings = RI_Util::update_coulomb_settings(this->info.coulomb_param, ucell, this->p_kv);

	bool init_MGT = true;
	for(const auto &settings_list : this->coulomb_settings)
	{
		this->exx_objs[settings_list.first].abfs_ccp = Conv_Coulomb_Pot_K::cal_orbs_ccp(
			this->abfs, settings_list.second.second, ccp_rmesh_times_modified);
		this->exx_objs[settings_list.first].cv.set_orbitals(ucell, orb,
															this->lcaos, this->abfs, this->exx_objs[settings_list.first].abfs_ccp,
															this->info.kmesh_times, this->MGT, init_MGT, settings_list.second.first );
		init_MGT = false; // only init once
		if (settings_list.first == Conv_Coulomb_Pot_K::Coulomb_Method::Ewald)
		{
			const auto &coulomb_param = settings_list.second.second;
			if (this->info.rotate_abfs == true)
			{
				// Fit Gaussian to original abfs and rotate with same coefficients
				ModuleBase::TITLE("fit_gaussian_before_rotate_start");
				std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs;
				g_abfs = this->exx_objs[settings_list.first].evq.init_gauss(abfs_original);
				ModuleBase::TITLE("fit_gaussian_before_rotate_end");

				ModuleBase::TITLE("rotate_gaussian_start");
				// moment_abfs->rotate_abfs(g_abfs);
				ModuleBase::TITLE("rotate_gaussian_end");

				// Pass both abfs (rotated non-Gaussian) and g_abfs (Gaussian-fitted and rotated)
				this->exx_objs[settings_list.first].evq.init(ucell, orb,
															 this->mpi_comm, this->p_kv, this->lcaos, this->abfs, g_abfs,
															 coulomb_param, this->MGT, this->info.ccp_rmesh_times, this->info.kmesh_times);
			}
			else
			{
				this->exx_objs[settings_list.first].evq.init(ucell, orb,
															 this->mpi_comm, this->p_kv, this->lcaos, this->abfs,
															 coulomb_param, this->MGT, this->info.ccp_rmesh_times, this->info.kmesh_times);
			}
		}
	}

	ModuleBase::timer::tick("Exx_LRI", "init");
}

template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_ions(const UnitCell& ucell,
								  const bool write_cv)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_ions");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_ions");

	// init_radial_table_ions( cal_atom_centres_core(atom_pairs_core_origin), atom_pairs_core_origin );

	// this->m_abfsabfs.init_radial_table(Rradial);
	// this->m_abfslcaos_lcaos.init_radial_table(Rradial);

	std::vector<TA> atoms(ucell.nat);
	for(int iat=0; iat<ucell.nat; ++iat)
		{ atoms[iat] = iat; }
	std::map<TA,TatomR> atoms_pos;
	for(int iat=0; iat<ucell.nat; ++iat)
		{ atoms_pos[iat] = RI_Util::Vector3_to_array3( ucell.atoms[ ucell.iat2it[iat] ].tau[ ucell.iat2ia[iat] ] ); }
	const std::array<TatomR,Ndim> latvec
		= {RI_Util::Vector3_to_array3(ucell.a1),
		   RI_Util::Vector3_to_array3(ucell.a2),
		   RI_Util::Vector3_to_array3(ucell.a3)};
	const std::array<Tcell,Ndim> period = {this->p_kv->nmp[0], this->p_kv->nmp[1], this->p_kv->nmp[2]};

	this->exx_lri.set_parallel(this->mpi_comm, atoms_pos, latvec, period);

	// std::max(3) for gamma_only, list_A2 should contain cell {-1,0,1}. In the future distribute will be neighbour.
	const std::array<Tcell,Ndim> period_Vs = LRI_CV_Tools::cal_latvec_range<Tcell>(1+this->info.ccp_rmesh_times, ucell, orb_cutoff_);
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
		list_As_Vs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Vs, 2, false);

	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Vs;
	std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>> dVs;
	for(const auto &settings_list : this->coulomb_settings)
	{
		std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
			Vs_temp = this->exx_objs[settings_list.first].cv.cal_Vs(ucell,
				list_As_Vs.first, list_As_Vs.second[0],
				{{"writable_Vws",true}});
		this->exx_objs[settings_list.first].cv.Vws = LRI_CV_Tools::get_CVws(ucell,Vs_temp);
		if (settings_list.first == Conv_Coulomb_Pot_K::Coulomb_Method::Ewald && this->info.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Ccp)
		{
			this->exx_objs[settings_list.first].evq.init_ions(ucell, period_Vs);
			const auto &coulomb_param = settings_list.second.second;
			std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_ewald;
			for(const auto &param_list : coulomb_param)
			{
				std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_ewald_temp;
				switch(param_list.first)
				{
					case Conv_Coulomb_Pot_K::Coulomb_Type::Fock:
					{
						double chi = this->exx_objs[settings_list.first].evq.get_singular_chi(ucell, param_list.second, 2.0);
						Vs_ewald_temp =  this->exx_objs[settings_list.first].evq.cal_Vs(ucell, chi, Vs_temp);
						break;
					}
					default:
					{
						throw std::invalid_argument( std::string(__FILE__) + " line " + std::to_string(__LINE__) );
					}
				}
				// Vs_temp cannot be covered here
				Vs_ewald = Vs_ewald.empty() ? Vs_ewald_temp : LRI_CV_Tools::add(Vs_ewald, Vs_ewald_temp);
			}
			Vs_temp = Vs_ewald;
		}
		Vs = Vs.empty() ? Vs_temp : LRI_CV_Tools::add(Vs, Vs_temp);

		if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
		{
			std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>>
				dVs_temp = this->exx_objs[settings_list.first].cv.cal_dVs(ucell,
					list_As_Vs.first, list_As_Vs.second[0],
					{{"writable_dVws",true}});
			this->exx_objs[settings_list.first].cv.dVws = LRI_CV_Tools::get_dCVws(ucell,dVs_temp);
			dVs = dVs.empty() ? dVs_temp : LRI_CV_Tools::add(dVs, dVs_temp);
		}
	}
	if (write_cv && GlobalV::MY_RANK == 0)
		{ LRI_CV_Tools::write_Vs_abf(Vs, PARAM.globalv.global_out_dir + "Vs"); }
	this->exx_lri.set_Vs(std::move(Vs), this->info.V_threshold);

	if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
	{
		std::array<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>, Ndim>
			dVs_order = LRI_CV_Tools::change_order(std::move(dVs));
		this->exx_lri.set_dVs(std::move(dVs_order), this->info.V_grad_threshold);
		if(PARAM.inp.cal_stress)
		{
			std::array<std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>,3> dVRs = LRI_CV_Tools::cal_dMRs(ucell,dVs_order);
			this->exx_lri.set_dVRs(std::move(dVRs), this->info.V_grad_R_threshold);
		}
	}

	const std::array<Tcell,Ndim> period_Cs = LRI_CV_Tools::cal_latvec_range<Tcell>(2, ucell,orb_cutoff_);
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
		list_As_Cs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Cs, 2, false);

	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Cs;
	std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>> dCs;
	for(const auto &settings_list : this->coulomb_settings)
	{
		if(settings_list.second.first)
		{
			std::pair<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>,
				std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>>>
					Cs_dCs = this->exx_objs[settings_list.first].cv.cal_Cs_dCs(
						ucell,
						list_As_Cs.first, list_As_Cs.second[0],
						{{"cal_dC",PARAM.inp.cal_force||PARAM.inp.cal_stress},
						{"writable_Cws",true}, {"writable_dCws",true}, {"writable_Vws",false}, {"writable_dVws",false}});
			std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> &Cs_temp = std::get<0>(Cs_dCs);
			this->exx_objs[settings_list.first].cv.Cws = LRI_CV_Tools::get_CVws(ucell,Cs_temp);
			Cs = Cs.empty() ? Cs_temp : LRI_CV_Tools::add(Cs, Cs_temp);

			if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
			{
				std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>> &dCs_temp = std::get<1>(Cs_dCs);
				this->exx_objs[settings_list.first].cv.dCws = LRI_CV_Tools::get_dCVws(ucell,dCs_temp);
				dCs = dCs.empty() ? dCs_temp : LRI_CV_Tools::add(dCs, dCs_temp);
			}
		}
	}
	if (write_cv && GlobalV::MY_RANK == 0)
		{ LRI_CV_Tools::write_Cs_ao(Cs, PARAM.globalv.global_out_dir + "Cs"); }
	this->exx_lri.set_Cs(std::move(Cs), this->info.C_threshold);

	if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
	{
		std::array<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>, Ndim>
			dCs_order = LRI_CV_Tools::change_order(std::move(dCs));
		this->exx_lri.set_dCs(std::move(dCs_order), this->info.C_grad_threshold);
		if(PARAM.inp.cal_stress)
		{
			std::array<std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>,3> dCRs = LRI_CV_Tools::cal_dMRs(ucell,dCs_order);
			this->exx_lri.set_dCRs(std::move(dCRs), this->info.C_grad_R_threshold);
		}
	}
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_ions");
}

template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_ions_rpa(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_cut,
                                      std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Cs,
                                      const UnitCell& ucell,
                                      const bool write_cv)
{
	ModuleBase::TITLE("Exx_LRI", "cal_exx_ions_rpa");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_ions_rpa");

	std::vector<TA> atoms(ucell.nat);
	for(int iat=0; iat<ucell.nat; ++iat)
		atoms[iat] = iat;
	std::map<TA,TatomR> atoms_pos;
	for(int iat=0; iat<ucell.nat; ++iat)
		atoms_pos[iat] = RI_Util::Vector3_to_array3( ucell.atoms[ucell.iat2it[iat]].tau[ucell.iat2ia[iat]] );
	const std::array<TatomR,Ndim> latvec
		= {RI_Util::Vector3_to_array3(ucell.a1),
		   RI_Util::Vector3_to_array3(ucell.a2),
		   RI_Util::Vector3_to_array3(ucell.a3)};
	const std::array<Tcell,Ndim> period = {this->p_kv->nmp[0], this->p_kv->nmp[1], this->p_kv->nmp[2]};

	this->exx_lri.set_parallel(this->mpi_comm, atoms_pos, latvec, period);

	// std::max(3) for gamma_only, list_A2 should contain cell {-1,0,1}. In the future distribute will be neighbour.
	const std::array<Tcell,Ndim> period_Vs = LRI_CV_Tools::cal_latvec_range<Tcell>(1+this->info.ccp_rmesh_times, ucell, orb_cutoff_);
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
		list_As_Vs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Vs, 2, false);

	// Find the first Coulomb method (RPA typically uses only one)
	auto first_setting = this->coulomb_settings.begin();
	const auto &coulomb_method = first_setting->first;
	const auto &coulomb_param_map = first_setting->second.second;

	// Calculate Vs for the first (and typically only) Coulomb method
	Vs_cut = this->exx_objs[coulomb_method].cv.cal_Vs(ucell,
		list_As_Vs.first, list_As_Vs.second[0],
		{{"writable_Vws",true},{"use_moment",true}});
	this->exx_objs[coulomb_method].cv.Vws = LRI_CV_Tools::get_CVws(ucell,Vs_cut);

	int flag = 0;
	for(const auto& IJRc: this->exx_objs[coulomb_method].cv.Vws)
	{
		const TA& I = IJRc.first;
		const auto& JRc = IJRc.second;
		for(const auto& JRc_tensor: JRc)
		{
			const TA& J = JRc_tensor.first;
			const auto Rc = JRc_tensor.second;
			for(const auto& Rc_tensor: Rc)
			{
				const auto& R = Rc_tensor.first;
				flag += 1;
			}
		}
	}
	std::cout << "Inside No. VIJR =" << flag << std::endl;

	// Check if Hf Coulomb type is used
	bool has_hf = false;
	for(const auto& param_list : coulomb_param_map)
	{
		if(param_list.first == Conv_Coulomb_Pot_K::Coulomb_Type::Hf)
		{
			has_hf = true;
			break;
		}
	}

	if(this->info.coul_moment == true && has_hf)
	{
		double hf_Rcut = std::pow(0.75 * this->p_kv->get_nkstot_full() * ucell.omega / (ModuleBase::PI), 1.0/3.0);
		// To cal Cs, we still cal all Vs(R) in r space
		moment_abfs->cal_VR(ucell, this->abfs, list_As_Vs, orb_cutoff_, hf_Rcut, this->exx_objs[coulomb_method].cv, Vs_cut);
		delete moment_abfs;
		moment_abfs = nullptr;
		malloc_trim(0);
	}

	flag = 0;
	for(const auto& IJRc: this->exx_objs[coulomb_method].cv.Vws)
	{
		const auto& JRc = IJRc.second;
		for(const auto& JRc_tensor: JRc)
		{
			const auto Rc = JRc_tensor.second;
			for(const auto& Rc_tensor: Rc)
			{
				flag += 1;
			}
		}
	}
	std::cout << "All No. VIJR =" << flag << std::endl;

	// Handle Ewald method with Ccp Coulomb type
	if(coulomb_method == Conv_Coulomb_Pot_K::Coulomb_Method::Ewald &&  && this->info.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Ccp)
	{
		this->exx_objs[coulomb_method].evq.init_ions(ucell, period_Vs);

		std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Vs_ewald;
		for(const auto &param_list : coulomb_param_map)
		{
			std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Vs_ewald_temp;
			switch(param_list.first)
			{
				case Conv_Coulomb_Pot_K::Coulomb_Type::Ccp:
				{
					Singular_Value::Fq_type fq_type = Singular_Value::Fq_type::plain;
					if(auto it = param_list.second.find("fq_type"); it != param_list.second.end())
					{
						if(it->second == "standard") fq_type = Singular_Value::Fq_type::standard;
						else if(it->second == "dyn") fq_type = Singular_Value::Fq_type::dyn;
					}
					double chi = this->exx_objs[coulomb_method].evq.get_singular_chi(ucell, fq_type, 2.0);
					Vs_ewald_temp = this->exx_objs[coulomb_method].evq.cal_Vs(ucell, chi, Vs_cut);
					std::cout << "Use exx_ccp_rmesh_times=" << this->info.ccp_rmesh_times << " to calculate full Coulomb" << std::endl;
					break;
				}
				default:
				{
					throw std::invalid_argument( std::string(__FILE__) + " line " + std::to_string(__LINE__) );
				}
			}
			Vs_ewald = Vs_ewald.empty() ? Vs_ewald_temp : LRI_CV_Tools::add(Vs_ewald, Vs_ewald_temp);
		}
		Vs_cut = Vs_ewald;
	}

	if(write_cv && GlobalV::MY_RANK==0)
	{
		if(coulomb_method == Conv_Coulomb_Pot_K::Coulomb_Method::Ewald)
			LRI_CV_Tools::write_Vs_abf(Vs_cut, PARAM.globalv.global_out_dir + "Vs_full");
		else
			LRI_CV_Tools::write_Vs_abf(Vs_cut, PARAM.globalv.global_out_dir + "Vs_cut");
	}
	this->exx_lri.set_Vs(std::move(Vs_cut), this->info.V_threshold);

	if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
	{
		std::map<TA,std::map<TAC,std::array<RI::Tensor<Tdata>,Ndim>>> dVs
			= this->exx_objs[coulomb_method].cv.cal_dVs(ucell,
				list_As_Vs.first, list_As_Vs.second[0],
				{{"writable_dVws",true}});
		this->exx_objs[coulomb_method].cv.dVws = LRI_CV_Tools::get_dCVws(ucell,dVs);

		std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,Ndim> dVs_order
			= LRI_CV_Tools::change_order(std::move(dVs));
		this->exx_lri.set_dVs(std::move(dVs_order), this->info.V_grad_threshold);
		if(PARAM.inp.cal_stress)
		{
			std::array<std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>,3> dVRs
				= LRI_CV_Tools::cal_dMRs(ucell,dVs_order);
			this->exx_lri.set_dVRs(std::move(dVRs), this->info.V_grad_R_threshold);
		}
	}

	const std::array<Tcell,Ndim> period_Cs = LRI_CV_Tools::cal_latvec_range<Tcell>(2, ucell,orb_cutoff_);
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
		list_As_Cs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Cs, 2, false);
	std::pair<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,
			  std::map<TA,std::map<TAC,std::array<RI::Tensor<Tdata>,3>>>>
		Cs_dCs = this->exx_objs[coulomb_method].cv.cal_Cs_dCs(ucell,
			list_As_Cs.first, list_As_Cs.second[0],
			{{"cal_dC",PARAM.inp.cal_force||PARAM.inp.cal_stress},
			 {"writable_Cws",true},
			 {"writable_dCws",true},
			 {"writable_Vws",false},
			 {"writable_dVws",false}});
	Cs = std::get<0>(Cs_dCs);
	this->exx_objs[coulomb_method].cv.Cws = LRI_CV_Tools::get_CVws(ucell,Cs);
	if(write_cv && GlobalV::MY_RANK==0)
	{
		LRI_CV_Tools::write_Cs_ao(Cs, PARAM.globalv.global_out_dir + "Cs");
	}
	this->exx_lri.set_Cs(Cs, this->info.C_threshold);

	if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
	{
		std::map<TA,std::map<TAC,std::array<RI::Tensor<Tdata>,3>>>& dCs = std::get<1>(Cs_dCs);
		this->exx_objs[coulomb_method].cv.dCws = LRI_CV_Tools::get_dCVws(ucell,dCs);
		std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,Ndim> dCs_order
			= LRI_CV_Tools::change_order(std::move(dCs));
		this->exx_lri.set_dCs(std::move(dCs_order), this->info.C_grad_threshold);
		if(PARAM.inp.cal_stress)
		{
			std::array<std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>,3> dCRs
				= LRI_CV_Tools::cal_dMRs(ucell,dCs_order);
			this->exx_lri.set_dCRs(std::move(dCRs), this->info.C_grad_R_threshold);
		}
	}
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_ions_rpa");
}

template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_elec(const std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>& Ds,
	const UnitCell& ucell,
	const Parallel_Orbitals& pv,
	const ModuleSymmetry::Symmetry_rotation* p_symrot)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_elec");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_elec");

	const std::vector<std::tuple<std::set<TA>, std::set<TA>>> judge = RI_2D_Comm::get_2D_judge(ucell,pv);

	if(p_symrot)
		{ this->exx_lri.set_symmetry(true, p_symrot->get_irreducible_sector()); }
	else
		{ this->exx_lri.set_symmetry(false, {}); }

	this->Hexxs.resize(PARAM.inp.nspin);
	this->Eexx = 0;
	for(int is=0; is<PARAM.inp.nspin; ++is)
	{
		const std::string suffix = ((PARAM.inp.cal_force || PARAM.inp.cal_stress) ? std::to_string(is) : "");

		this->exx_lri.set_Ds(Ds[is], this->info.dm_threshold, suffix);
		this->exx_lri.cal_Hs({ "","",suffix });

		if (!p_symrot)
		{
			this->Hexxs[is] = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
				this->mpi_comm, std::move(this->exx_lri.Hs), std::get<0>(judge[is]), std::get<1>(judge[is]));
		}
		else
		{
			// reduce but not repeat
			auto Hs_a2D = this->exx_lri.post_2D.set_tensors_map2(this->exx_lri.Hs);
			// rotate locally without repeat
			Hs_a2D = p_symrot->restore_HR(ucell.symm, ucell.atoms, ucell.st, 'H', Hs_a2D);
			// cal energy using full Hs without repeat
			this->exx_lri.energy = this->exx_lri.post_2D.cal_energy(
				this->exx_lri.post_2D.saves["Ds_" + suffix],
				this->exx_lri.post_2D.set_tensors_map2(Hs_a2D));
			// get repeated full Hs for abacus
			this->Hexxs[is] = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
				this->mpi_comm, std::move(Hs_a2D), std::get<0>(judge[is]), std::get<1>(judge[is]));
		}
		this->Eexx += std::real(this->exx_lri.energy);
		post_process_Hexx(this->Hexxs[is]);
	}
	this->Eexx = post_process_Eexx(this->Eexx);
	this->exx_lri.set_symmetry(false, {});
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_elec");
}

template<typename Tdata>
void Exx_LRI<Tdata>::post_process_Hexx( std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> &Hexxs_io ) const
{
	ModuleBase::TITLE("Exx_LRI","post_process_Hexx");
	constexpr Tdata frac = -1 * 2;								// why?	Hartree to Ry?
	const std::function<void(RI::Tensor<Tdata>&)>
		multiply_frac = [&frac](RI::Tensor<Tdata> &t)
		{ t = t*frac; };
	RI::Map_Operator::for_each( Hexxs_io, multiply_frac );
}

template<typename Tdata>
double Exx_LRI<Tdata>::post_process_Eexx(const double& Eexx_in) const
{
	ModuleBase::TITLE("Exx_LRI","post_process_Eexx");
	const double SPIN_multiple = std::map<int, double>{ {1,2}, {2,1}, {4,1} }.at(PARAM.inp.nspin);				// why?
	const double frac = -SPIN_multiple;
	return frac * Eexx_in;
}

/*
post_process_old
{
	// D
	const std::map<int,double> SPIN_multiple = {{1,0.5}, {2,1}, {4,1}};							// ???
	DR *= SPIN_multiple.at(NSPIN);

	// H
	HR *= -2;

	// E
	const std::map<int,double> SPIN_multiple = {{1,2}, {2,1}, {4,1}};							// ???
	energy *= SPIN_multiple.at(PARAM.inp.nspin);			// ?
	energy /= 2;					// /2 for Ry
}
*/

template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_force(const int& nat)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_force");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_force");

	this->force_exx.create(nat, Ndim);
	for(int is=0; is<PARAM.inp.nspin; ++is)
	{
		this->exx_lri.cal_force({"","",std::to_string(is),"",""});
		for(std::size_t idim=0; idim<Ndim; ++idim) {
			for(const auto &force_item : this->exx_lri.force[idim]) {
				this->force_exx(force_item.first, idim) += std::real(force_item.second);
					} 		}
	}

	const double SPIN_multiple = std::map<int,double>{{1,2}, {2,1}, {4,1}}.at(PARAM.inp.nspin);				// why?
	const double frac = -2 * SPIN_multiple;		// why?
	this->force_exx *= frac;
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_force");
}


template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_stress(const double& omega, const double& lat0)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_stress");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_stress");

	this->stress_exx.create(Ndim, Ndim);
	for(int is=0; is<PARAM.inp.nspin; ++is)
	{
		this->exx_lri.cal_stress({"","",std::to_string(is),"",""});
		for(std::size_t idim0=0; idim0<Ndim; ++idim0) {
			for(std::size_t idim1=0; idim1<Ndim; ++idim1) {
				this->stress_exx(idim0,idim1) += std::real(this->exx_lri.stress(idim0,idim1));
				} 	}
	}

	const double SPIN_multiple = std::map<int,double>{{1,2}, {2,1}, {4,1}}.at(PARAM.inp.nspin);				// why?
	const double frac = 2 * SPIN_multiple / omega * lat0;		// why?
	this->stress_exx *= frac;

	ModuleBase::timer::tick("Exx_LRI", "cal_exx_stress");
}

/*
template<typename Tdata>
std::vector<std::vector<int>> Exx_LRI<Tdata>::get_abfs_nchis() const
{
	std::vector<std::vector<int>> abfs_nchis;
	for (const auto& abfs_T : this->abfs)
	{
		std::vector<int> abfs_nchi_T;
		for (const auto& abfs_L : abfs_T)
			{ abfs_nchi_T.push_back(abfs_L.size()); }
		abfs_nchis.push_back(abfs_nchi_T);
	}
	return abfs_nchis;
}
*/

#endif
