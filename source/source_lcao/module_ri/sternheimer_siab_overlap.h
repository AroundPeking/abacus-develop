#ifndef STERNHEIMER_SIAB_OVERLAP_H
#define STERNHEIMER_SIAB_OVERLAP_H

#include "source_lcao/module_ri/sternheimer_fd_hamiltonian.h"

#include <complex>
#include <vector>

namespace module_ri
{
namespace sternheimer_siab
{

double norm(const std::vector<std::complex<double>>& y, double delta_omega);

std::vector<std::complex<double>> overlap_q(
    const std::vector<std::complex<double>>& y,
    const std::vector<std::vector<std::complex<double>>>& primitives,
    double delta_omega);

std::vector<std::complex<double>> overlap_s(
    const std::vector<std::vector<std::complex<double>>>& primitives,
    double delta_omega);

std::vector<std::vector<std::complex<double>>> perturbation_matrices(
    const std::vector<std::vector<std::complex<double>>>& basis_functions,
    const std::vector<std::vector<double>>& potentials_ha,
    double delta_omega,
    int* threads_used = nullptr);

std::vector<std::complex<double>> hamiltonian_matrix(
    const std::vector<std::vector<std::complex<double>>>& basis_functions,
    const ModuleRI::SternheimerFDHamiltonian& hamiltonian,
    double delta_omega,
    int column_batch_size = 64);

std::vector<std::complex<double>> overlap_q_reciprocal(
    const std::vector<std::complex<double>>& response_coefficients,
    const std::vector<std::vector<std::complex<double>>>& primitive_coefficients);

std::vector<std::complex<double>> overlap_q_reciprocal_contiguous(
    const std::vector<std::complex<double>>& response_coefficients,
    const std::vector<std::complex<double>>& primitive_coefficients,
    int primitive_count,
    int reciprocal_count);

std::vector<std::complex<double>> overlap_s_reciprocal(
    const std::vector<std::vector<std::complex<double>>>& primitive_coefficients);

} // namespace sternheimer_siab
} // namespace module_ri

#endif
