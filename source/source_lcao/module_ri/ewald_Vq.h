//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_H
#define EWALD_VQ_H

#include <array>
#include <map>

#include "gaussian_abfs.h"

template <typename Tdata>
class Ewald_Vq
{
  private:
    using TA = int;
    using TC = std::array<int, 3>;
    using TAC = std::pair<TA, TC>;

  public:
    void init(std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos,
              std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
              const double& gauss_gamma);

  private:
    LRI_CV<Tdata> cv;
    Gaussian_Abfs gaussian_abfs;
    const double gamma;

    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_lcaos;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> g_abfs_ccp;

    const int nspin0 = std::map<int, int>{
        {1, 1},
        {2, 2},
        {4, 1}
    }.at(GlobalV::NSPIN);

  private:
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> init_gauss(
        std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in);
}
#include "ewald_Vq.hpp"

#endif