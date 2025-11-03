
#ifndef EXX_ROTATE_ABFS_H
#define EXX_ROTATE_ABFS_H

#include "LRI_CV.h"
// #include "module_xc/exx_info.h"
// #include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_base/matrix.h"
#include "module_ri/Exx_LRI.h"
// #include "module_ri/Exx_LRI.h"
// #include <RI/physics/Exx.h>
#include <RI/ri/RI_Tools.h>
#include <array>
#include <map>
#include <mpi.h>
#include <vector>

template <typename Tdata>
class Moment_abfs
{
  private:
    using TA = int;
    using Tcell = int;
    static constexpr std::size_t Ndim = 3;
    using TC = std::array<Tcell, Ndim>;
    using Tq = std::array<double, Ndim>;
    using TAC = std::pair<TA, TC>;
    using TAq = std::pair<TA, Tq>;

  public:
    Moment_abfs(Exx_Info::Exx_Info_RI& info_in) : info(info_in) {};
    ~Moment_abfs() {};
    void cal_VR(
        const UnitCell& ucell,
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
        const ORB_gaunt_table& MGT,
        const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>& list_r);
    std::vector<std::vector<std::vector<double>>> cal_multipole(
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in);
    double cal_clmlm(int l2, int m2, int l, int m, const ORB_gaunt_table& MGT) const;
    /// double factorial
    int dfact(const int& l) const;
    int factorial(const int& n) const;
    void diverge_list(
        const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>& list_As_Vs,
        std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>& list_k,
        std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>& list_r,
        const UnitCell& ucell,
        const std::vector<double>& orb_cutoff);
    void merge_list(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_cut);

  private:
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> VR;
    Exx_Info::Exx_Info_RI& info;
};
#include "exx_rotate_abfs.hpp"

#endif
