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
    std::pair<RI::Tensor<std::complex<double>>, std::vector<RI::Tensor<std::complex<double>>>> get_Vq(
        const int& lp_max,
        const int& lq_max, // Maximum L for which to calculate interaction.
        const ModuleBase::Vector3<double>& qvec,
        const ModuleBase::Matrix3& G,
        const double& chi, // Singularity corrected value at q=0.
        const double& lambda,
        const ModuleBase::Vector3<double>& tau,
        const ORB_gaunt_table& MGT,
                        const bool& cal_deriv);

    /*
Calculate the lattice sum over a Gaussian:
  S(k) := \sum_G |k+G|^{power+L} \exp(-lambda*|k+G|^2) Y_{LM}(k+G) * \exp(i(k+G)\tau)
  d_S(k) := S(k) * i * (k+G)
*/
    static std::pair<std::vector<std::complex<double>>, std::vector<std::vector<std::complex<double>>>>
        get_lattice_sum(const ModuleBase::Vector3<double>& qvec,
                        const ModuleBase::Matrix3& G,
                        const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
                        const double& exponent,
                        const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
                        const int& lmax,           // Maximum angular momentum the sum is needed for.
                        const ModuleBase::Vector3<double>& tau,
                        const bool& cal_deriv);

    // construct gaussian basis based on original NAO
    Numerical_Orbital_Lm Gauss(const Numerical_Orbital_Lm& orb, const double& lambda);

  private:
    // calculates the double factorial n!! of n
    static double double_factorial(const int& n);
    static std::vector<int> get_n_supercells(const ModuleBase::Matrix3& G, const double& Gmax);
};

#endif