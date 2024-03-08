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

class Gaussian_Abfs
{
  public:
    Gaussian_Abfs(const ORB_gaunt_table& MGT_in);

  private:
    ORB_gaunt_table MGT;

  public:
    RI::Tensor<std::complex<double>> get_Vq(const int& lp_max,
                                            const int& lq_max, // Maximum L for which to calculate interaction.
                                            const std::vector<ModuleBase::Vector3<double>>& gk,
                                            const double& chi, // Singularity corrected value at q=0.
                                            const double& gamma,
                                            const ModuleBase::Vector3<double>& tau);

  private:
    // construct gaussian basis based on original NAO
    Numerical_Orbital_Lm Gauss(const Numerical_Orbital_Lm& orb, const double& gamma);

    // calculates the double factorial n!! of n
    static double double_factorial(const int& n);

    /*
    Calculate the lattice sum over a Gaussian:
      S(k) := \sum_G |k+G|^{power+L} \exp(-gamma*|k+G|^2) Y_{LM}(k+G) * \exp(i(k+G)\tau)
    */
    static std::vector<std::complex<double>> get_lattice_sum(
        const std::vector<ModuleBase::Vector3<double>>& gk,
        const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
        const double& gamma,
        const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
        const int& lmax,           // Maximum angular momentum the sum is needed for.
        const ModuleBase::Vector3<double>& tau);
}

#endif