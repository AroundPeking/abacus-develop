
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

class Parallel_Orbitals;
class K_Vectors;

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
    void cal_large_Cs(const UnitCell& ucell, const LCAO_Orbitals& orb, const K_Vectors& kv);
    void cal_postSCF_exx(const elecstate::DensityMatrix<T, Tdata>& dm,
                         const MPI_Comm& mpi_comm_in,
                         const UnitCell& ucell,
                         const K_Vectors& kv,
                         const LCAO_Orbitals& orb);

  private:
    Exx_Info::Exx_Info_RI& info;
    const K_Vectors* p_kv = nullptr;
    MPI_Comm mpi_comm;
    double exx_ccp_rmesh_times;

    std::vector<double> orb_cutoff_;

    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> lcaos;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_ccp;
    // shrinked abfs
    ORB_gaunt_table MGT;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_s;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_s_ccp;

    // Exx_LRI<double> exx_postSCF_double(info);
    // LRI_CV<Tdata> cv;
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs_period;
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Cs_period;
};
#include "exx_rotate_abfs.hpp"

#endif
