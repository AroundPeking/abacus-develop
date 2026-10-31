//==========================================================
// AUTHOR : mohan
// DATE : 2009-4-2
// Last Modify: 2009-08-28
//==========================================================
#ifndef NUMERICAL_BASIS_H
#define NUMERICAL_BASIS_H
#include <complex>
#include <string>
#include <vector>

#include "bessel_basis.h"
#include "../../source_base/complexarray.h"
#include "../../source_base/complexmatrix.h"
#include "../../source_base/global_function.h"
#include "../../source_base/intarray.h"
#include "../../source_base/matrix.h"
#include "../../source_base/vector3.h"
#include "../../source_basis/module_pw/pw_basis_k.h"
#include "../../source_cell/klist.h"
#include "source_cell/stru_fac.h"
#include "../../source_psi/psi.h"
//==========================================================
// CLASS :
// NAME :  Numerical_Basis
//==========================================================
class Numerical_Basis
{
  public:
    // this is not a good idea to construct the instance. Instead, there are many variables that CAN DEFINE the identity of this instance,
    // these parameters should be provided in the constructor. For future refactor, I list them here:
    // bessel_nao_*: including ecut, rcut, smearing, sigma, etc.
    // a file name: for the function start_from_file_k, if starts from file, can construct a different instance, instead of using the same instance.
    Numerical_Basis();
    ~Numerical_Basis();

    /// Numerical-Bessel (SIAB) primitive parameters carried through the
    /// SIAB export path. They mirror the bessel_nao_* input settings that the
    /// existing spillage path already uses.
    struct SIABPrimitiveParameters
    {
        SIABPrimitiveParameters()
            : ecut_ry(0.0), rcut_bohr(0.0), smooth(false), sigma(0.0), tolerance(0.0), lmax(-1)
        {
        }

        SIABPrimitiveParameters(const double ecut_ry_in,
                                const double rcut_bohr_in,
                                const bool smooth_in,
                                const double sigma_in,
                                const double tolerance_in,
                                const int lmax_in = -1)
            : ecut_ry(ecut_ry_in),
              rcut_bohr(rcut_bohr_in),
              smooth(smooth_in),
              sigma(sigma_in),
              tolerance(tolerance_in),
              lmax(lmax_in)
        {
        }

        double ecut_ry;
        double rcut_bohr;
        bool smooth;
        double sigma;
        double tolerance;
        int lmax;
    };

    /// Rank-local SIAB primitive values sampled on the real-space FFT grid.
    struct SIABPrimitiveGridBlock
    {
        int type_index;
        std::string element;
        int atom_index;
        int l;
        int m;
        int n_primitive;
        int offset;
        /// Rank-local values ordered by primitive index: values[ie][local_grid_index].
        std::vector<std::vector<std::complex<double>>> values;
    };

    /// Physically normalized SIAB primitive coefficients in the PW basis.
    struct SIABPrimitiveReciprocalBlock
    {
        int type_index;
        std::string element;
        int atom_index;
        int l;
        int m;
        int n_primitive;
        int offset;
        /// Physically normalized PW coefficients ordered by primitive index.
        std::vector<std::vector<std::complex<double>>> values;
    };

    /// Caller-provided numerical-Bessel settings for the SIAB primitive path.
    /// Passed explicitly so this module does not read process-global INPUT state.
    struct SIABPrimitiveInputs
    {
        SIABPrimitiveInputs()
            : ecut_spec("default"), ecutwfc(0.0), smooth(false), sigma(0.0), tolerance(0.0)
        {
        }

        /// bessel_nao_rcuts: candidate primitive cutoffs in Bohr.
        std::vector<double> rcuts;
        /// bessel_nao_ecut: either the literal "default" or a numeric cutoff string.
        std::string ecut_spec;
        /// ecutwfc: wavefunction cutoff in Ry, used when ecut_spec is "default".
        double ecutwfc;
        /// bessel_nao_smooth / bessel_nao_sigma / bessel_nao_tolerence.
        bool smooth;
        double sigma;
        double tolerance;
    };

    /// Return the ABACUS real-spherical-harmonic index for a conventional m.
    static int siab_abacus_m(const int conventional_m);

    /// Build the primitive parameters from explicitly supplied numerical-Bessel settings.
    static SIABPrimitiveParameters siab_parameters_from_input(const SIABPrimitiveInputs& settings,
                                                              const int rcut_index,
                                                              const int lmax = -1);

    /// Build rank-local, physically normalized SIAB primitives on the PW FFT grid.
    std::vector<SIABPrimitiveGridBlock> siab_primitive_grid_values(
        const int ik,
        const ModulePW::PW_Basis_K* wfcpw,
        const Structure_Factor& sf,
        const UnitCell& ucell,
        const SIABPrimitiveParameters& parameters);

    std::vector<SIABPrimitiveReciprocalBlock> siab_primitive_reciprocal_values(
        int ik,
        const ModulePW::PW_Basis_K* wfcpw,
        const Structure_Factor& sf,
        const UnitCell& ucell,
        const SIABPrimitiveParameters& parameters);

    // void start_from_file_k(const int& ik, ModuleBase::ComplexMatrix& psi, const Structure_Factor& sf, const ModulePW::PW_Basis_K* wfcpw, const UnitCell& ucell);
    void output_overlap(const psi::Psi<std::complex<double>>& psi,
                        const Structure_Factor& sf,
                        const K_Vectors& kv,
                        const ModulePW::PW_Basis_K* wfcpw,
                        const UnitCell& ucell,
                        const int& index);

  private:
    bool init_label = false;
    SIABPrimitiveParameters initialized_siab_parameters = {0.0, 0.0, false, 0.0, 0.0, -1};
    int initialized_siab_ntype = 0;
    int initialized_siab_lmax = 0;
    int initialized_siab_nmax = 0;
    std::vector<int> initialized_siab_basis_shape;

    Bessel_Basis bessel_basis;

    std::vector<ModuleBase::IntArray> mu_index;
    static std::vector<ModuleBase::IntArray> init_mu_index(const UnitCell& ucell);
    static std::vector<int> siab_basis_shape_signature(const UnitCell& ucell);

    void initialize_siab_basis(const UnitCell& ucell, const SIABPrimitiveParameters& parameters);

    static void fill_reciprocal_primitive(
        const int l,
        const int abacus_m,
        const int primitive_index,
        const std::complex<double>* structure_factor,
        const ModuleBase::realArray& flq,
        const ModuleBase::matrix& ylm,
        const std::vector<double>& gpow,
        const double normalization,
        std::vector<std::complex<double>>& values);

    void numerical_atomic_wfc(const int& ik,
                              const ModulePW::PW_Basis_K* wfcpw,
                              ModuleBase::ComplexMatrix& psi,
                              const Structure_Factor& sf,
                              const UnitCell& ucell);

    ModuleBase::ComplexArray cal_overlap_Q(const int& ik,
                                           const int& np,
                                           const ModulePW::PW_Basis_K* wfcpw,
                                           const psi::Psi<std::complex<double>>& psi,
                                           const double derivative_order,
                                           const Structure_Factor& sf,
                                           const UnitCell& ucell) const;

    // computed in the plane-wave basis
    ModuleBase::ComplexArray cal_overlap_Sq(const int& ik,
                                            const int& np,
                                            const double derivative_order,
                                            const Structure_Factor& sf,
                                            const ModulePW::PW_Basis_K* wfcpw,
                                            const UnitCell& ucell) const;

    static ModuleBase::matrix cal_overlap_V(const ModulePW::PW_Basis_K* wfcpw,
                                            const psi::Psi<std::complex<double>>& psi,
                                            const double derivative_order,
                                            const K_Vectors& kv,
                                            const double tpiba);

    // gk should be in the atomic unit (Bohr)
    ModuleBase::realArray cal_flq(const std::vector<ModuleBase::Vector3<double>> &gk,
                                  const int ucell_lmax) const;

    // Ylm does not depend on the magnitude so unit is not important
    static ModuleBase::matrix cal_ylm(const std::vector<ModuleBase::Vector3<double>> &gk,
                                      const int ucell_lmax);

    // gk and the returned gpow are both in the atomic unit (Bohr)
    static std::vector<double> cal_gpow(const std::vector<ModuleBase::Vector3<double>> &gk,
                                        const double derivative_order);

    static void output_info(std::ofstream& ofs, const Bessel_Basis& bessel_basis, const K_Vectors& kv, const UnitCell& ucell);

    static void output_k(std::ofstream& ofs, const K_Vectors& kv);

    static void output_overlap_Q(std::ofstream& ofs,
                                 const std::vector<ModuleBase::ComplexArray>& overlap_Q,
                                 const K_Vectors& kv);

    static void output_overlap_Sq(const std::string& name,
                                  std::ofstream& ofs,
                                  const std::vector<ModuleBase::ComplexArray>& overlap_Sq,
                                  const K_Vectors& kv);

    static void output_overlap_V(std::ofstream &ofs, const ModuleBase::matrix &overlap_V);
};

#endif
