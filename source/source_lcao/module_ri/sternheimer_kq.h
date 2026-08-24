#ifndef MODULE_RI_STERNHEIMER_KQ_H
#define MODULE_RI_STERNHEIMER_KQ_H

#include <array>
#include <complex>
#include <vector>

namespace ModuleRI
{

using SternheimerReducedKPoint = std::array<double, 3>;

struct SternheimerFoldedKPoint
{
    SternheimerReducedKPoint kpoint;
    std::array<int, 3> reciprocal_shift;
};

struct SternheimerKQPair
{
    int source_index;
    int target_index;
    SternheimerReducedKPoint folded_k_plus_q;
    std::array<int, 3> reciprocal_shift;
};

SternheimerFoldedKPoint fold_sternheimer_kpoint(const SternheimerReducedKPoint& kpoint);

double periodic_sternheimer_kpoint_distance(const SternheimerReducedKPoint& lhs, const SternheimerReducedKPoint& rhs);

// ABACUS folds real-space matrices with exp(+i 2 pi k.R).
std::complex<double> sternheimer_bloch_phase(const SternheimerReducedKPoint& kpoint,
                                             const std::array<int, 3>& lattice_translation);

// PW_Basis_K transforms the cell-periodic part u_k(r), while the FD solver
// stores the full Bloch function psi_k(r) with twisted boundary conditions.
std::vector<std::complex<double>> remove_sternheimer_bloch_phase(const std::vector<std::complex<double>>& bloch_values,
                                                                 int nx,
                                                                 int ny,
                                                                 int nz,
                                                                 const SternheimerReducedKPoint& kpoint);

std::vector<SternheimerKQPair> build_sternheimer_kq_map(const std::vector<SternheimerReducedKPoint>& kpoints,
                                                        const SternheimerReducedKPoint& qpoint,
                                                        double tolerance = 1.0e-10);

} // namespace ModuleRI

#endif
