
#ifndef EXX_ROTATE_ABFS_H
#define EXX_ROTATE_ABFS_H

#include "LRI_CV.h"
#include "module_esolver/esolver_ks_lcao.h"
// #include "module_xc/exx_info.h"
// #include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_base/matrix.h"
// #include "module_ri/Exx_LRI.h"
// #include <RI/physics/Exx.h>
#include <RI/ri/RI_Tools.h>
#include <array>
#include <map>
#include <mpi.h>
#include <vector>

template <typename T, typename Tdata>
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
    void init(const MPI_Comm& mpi_comm_in, const K_Vectors& kv_in, const std::vector<double>& orb_cutoff);
    std::vector<std::vector<std::vector<double>>> get_multipole(
        const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in);

  private:
    std::vector<std::vector<std::vector<double>>> multipole;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs;
};
#include "exx_rotate_abfs.hpp"

#endif
