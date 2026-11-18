#ifndef EXX_ABFS_CTOR_ORBS_H
#define EXX_ABFS_CTOR_ORBS_H

#include "exx_abfs.h"

#include <cstddef>
#include <limits>
#include <algorithm>
#include "source_cell/unitcell.h"
#include "source_basis/module_ao/orb_atomic_lm.h"

class LCAO_Orbitals;

class Exx_Abfs::Construct_Orbs
{
public:
	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> change_orbs( 
		const LCAO_Orbitals &orb_in,
		const double kmesh_times );
	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> change_orbs( 
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in,
		const double kmesh_times );

	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_same_atom( 
		const UnitCell &ucell,
		const LCAO_Orbitals& orb,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos,
		const double kmesh_times_mot,
		const double times_threshold=0);
		
	static void print_orbs_size(
		const UnitCell& ucell,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs,
		std::ostream &os);		

	// get the max number of orbitals among all elements
	// static int get_nmax_total(const
	// std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in); get
	// number of orbitals for each element static std::map<int, int>
	// get_nw(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>
	// &orb_in);

	// get multipole of orbitals for each element and angular moment
	static std::vector<std::vector<std::vector<double>>> get_multipole(
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in);

	static std::vector<double> get_Rcut(
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in);
	static inline double get_Rmax(const std::vector<double>& rcut)
	{
		return *std::max_element(rcut.begin(), rcut.end());
	}
	static inline double get_Rmax(
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orb_in)
	{
		std::vector<double> rcut = get_Rcut(orb_in);
		return get_Rmax(rcut);
	}
	template<typename T>
	static int get_Lmax(const std::vector<std::vector<T>> &orb)
	{
		// An empty channel list means "no angular momentum", so the answer is -1.
		// Writing it as max_element(...)->size() - 1 did not say that: size() is
		// unsigned, an all-empty input gave 0 - 1 = SIZE_MAX, and converting that
		// to int yielded -1 only by two's-complement luck. Callers then added two
		// such values (Matrix_Orbs11::init) and passed the result to
		// Center2_Orb::init_Table_Spherical_Bessel, which reached
		// Sph_Bessel_Recursive::cal_jlx with a negative lmax and aborted with
		// "jlx l<0". Filter empty tails first so the size comparison is meaningful.
		std::size_t lmax_size = 0;
		for (const auto &channels : orb)
		{
			lmax_size = std::max(lmax_size, channels.size());
		}
		return lmax_size == 0 ? -1 : static_cast<int>(lmax_size) - 1;
	}

	static void filter_empty_orbs(
		std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs);

  private:
	static std::vector<std::vector<std::vector<std::vector<double>>>> psi_mult_psi( 
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos );
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> psir_mult_psir( 
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos );
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> orth( 
		const std::vector<std::vector<std::vector<std::vector<double>>>> &psis,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &lcaos,
		const double norm_threshold = std::numeric_limits<double>::min() );

	static std::vector<std::vector<std::vector<std::vector<double>>>> pca(
		const UnitCell &ucell,
		const LCAO_Orbitals& orb,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &abfs,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs,
		const double kmesh_times_mot,
		const double times_threshold );
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> div_r( 
		const std::vector<std::vector<std::vector<std::vector<double>>>> &psirs,
		const std::vector<double> &r_radial );		
		
	static std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> orbital(
		const std::vector<std::vector<std::vector<std::vector<double>>>> &psis,
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs_info,
		const double kmesh_times);		
		
	static std::vector<std::vector<std::vector<std::vector<double>>>> get_psi(
		const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> &orbs );
};

#endif	// EXX_ABFS_IO_ASA_H
