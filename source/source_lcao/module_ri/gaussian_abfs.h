//=======================
// AUTHOR : jiyy
// DATE :   2024-02-27
//=======================

#ifndef GAUSSIAN_ABFS_H
#define GAUSSIAN_ABFS_H

#include <RI/global/Tensor.h>

#include <map>
#include <vector>

#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_basis/module_ao/ORB_gen_tables.h"
#include "module_basis/module_pw/pw_basis_k.h"

class Gaussian_Abfs
{
  public:
    RI::Tensor<std::complex<double>> get_Vq(const int& lp_max,
                                            const int& lq_max, // Maximum L for which to calculate interaction.
                                            const ModuleBase::Vector3<double>& qvec,
                                            const ModulePW::PW_Basis_K* wfc_basis,
                                            const double& chi, // Singularity corrected value at q=0.
                                            const double& lambda,
                                            const ModuleBase::Vector3<double>& tau);

    /*
Calculate the lattice sum over a Gaussian:
  S(k) := \sum_G |k+G|^{power+L} \exp(-lambda*|k+G|^2) Y_{LM}(k+G) * \exp(i(k+G)\tau)
*/
    static std::vector<std::complex<double>> get_lattice_sum(
        const ModuleBase::Vector3<double>& qvec,
        const ModulePW::PW_Basis_K* wfc_basis,
        const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
        const double& lambda,
        const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
        const int& lmax,           // Maximum angular momentum the sum is needed for.
        const ModuleBase::Vector3<double>& tau);

    // construct gaussian basis based on original NAO
    Numerical_Orbital_Lm Gauss(const Numerical_Orbital_Lm& orb, const double& lambda);

  private:
    ORB_gaunt_table MGT;

    // calculates the double factorial n!! of n
    static double double_factorial(const int& n);
};

#endif