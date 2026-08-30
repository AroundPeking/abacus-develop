#include "source_lcao/module_ri/sternheimer_periodic_solver.h"

#include <stdexcept>
#include <utility>

namespace ModuleRI
{

SternheimerPeriodicLinearResponse solve_sternheimer_periodic_linear_response(
    const bool use_delta_sternheimer,
    const SternheimerFDHamiltonian& hamiltonian,
    const std::vector<SternheimerFDHamiltonian::Vector>& occupied_wavefunctions,
    const double reference_eigenvalue,
    const SternheimerFDHamiltonian::Vector& rhs,
    const std::vector<SternheimerDeltaVirtualState>& virtual_states,
    const std::vector<SternheimerFDHamiltonian::Complex>& perturbation_matrix_elements,
    const double omega,
    const double volume_element,
    const SternheimerRPA::SolverOptions& options)
{
    SternheimerPeriodicLinearResponse result;
    result.projected_rhs = rhs;
    const auto dot = [volume_element](const SternheimerFDHamiltonian::Vector& lhs,
                                      const SternheimerFDHamiltonian::Vector& rhs_vector) {
        return sternheimer_fd_grid_dot(lhs, rhs_vector, volume_element);
    };
    SternheimerRPA::project_out_subspace(occupied_wavefunctions, dot, result.projected_rhs);
    if (use_delta_sternheimer)
    {
        SternheimerDeltaLinearResponse response
            = solve_delta_sternheimer_linear_response(hamiltonian,
                                                       occupied_wavefunctions,
                                                       reference_eigenvalue,
                                                       rhs,
                                                       virtual_states,
                                                       perturbation_matrix_elements,
                                                       omega,
                                                       volume_element,
                                                       options);
        result.wavefunction = response.response.reconstructed_wavefunction;
        result.solver = response.solver;
        result.residual_norm = response.residual_norm;
        result.full_grid_equation_residual_norm
            = sternheimer_fd_linear_response_residual_norm(hamiltonian,
                                                            occupied_wavefunctions,
                                                            reference_eigenvalue,
                                                            rhs,
                                                            result.wavefunction,
                                                            omega,
                                                            volume_element);
        result.hamiltonian_applications = response.hamiltonian_applications + 1;
        result.has_delta_components = true;
        result.delta_components = std::move(response.response);
        return result;
    }

    const SternheimerFDLinearResponse response
        = solve_sternheimer_fd_linear_response(hamiltonian,
                                                occupied_wavefunctions,
                                                reference_eigenvalue,
                                                rhs,
                                                omega,
                                                volume_element,
                                                options);
    result.wavefunction = response.delta_wavefunction;
    result.solver = response.solver;
    result.residual_norm = response.residual_norm;
    result.full_grid_equation_residual_norm = response.residual_norm;
    return result;
}

SternheimerPeriodicFrequencyRecyclingResult solve_sternheimer_periodic_frequency_recycling(
    const SternheimerFDHamiltonian& hamiltonian,
    const std::vector<SternheimerFDHamiltonian::Vector>& occupied_wavefunctions,
    const double reference_eigenvalue,
    const SternheimerFDHamiltonian::Vector& rhs,
    const std::vector<SternheimerDeltaVirtualState>& virtual_states,
    const std::vector<SternheimerFDHamiltonian::Complex>& perturbation_matrix_elements,
    const std::vector<double>& frequencies,
    const double volume_element,
    const SternheimerRPA::SolverOptions& solver_options,
    const SternheimerRPA::FrequencyRecyclingOptions& recycling_options)
{
    SternheimerPeriodicFrequencyRecyclingResult result;
    const auto dot = [volume_element](const SternheimerFDHamiltonian::Vector& lhs,
                                      const SternheimerFDHamiltonian::Vector& rhs_vector) {
        return sternheimer_fd_grid_dot(lhs, rhs_vector, volume_element);
    };
    SternheimerFDHamiltonian::Vector projected_rhs = rhs;
    SternheimerRPA::project_out_subspace(occupied_wavefunctions, dot, projected_rhs);

    const SternheimerDeltaFixedSubspace fixed_subspace
        = build_delta_sternheimer_fixed_subspace(occupied_wavefunctions, virtual_states);
    SternheimerDeltaFrequencyRecyclingResult delta_result
        = solve_delta_sternheimer_frequency_recycling(hamiltonian,
                                                       fixed_subspace,
                                                       reference_eigenvalue,
                                                       rhs,
                                                       virtual_states,
                                                       perturbation_matrix_elements,
                                                       frequencies,
                                                       volume_element,
                                                       solver_options,
                                                       recycling_options);
    result.recycling = std::move(delta_result.recycling);
    result.hamiltonian_applications = delta_result.hamiltonian_applications;
    result.responses.resize(delta_result.responses.size());
    for (std::size_t ifrequency = 0; ifrequency != delta_result.responses.size(); ++ifrequency)
    {
        SternheimerDeltaLinearResponse& delta_response = delta_result.responses[ifrequency];
        SternheimerPeriodicLinearResponse& response = result.responses[ifrequency];
        response.projected_rhs = projected_rhs;
        response.wavefunction = delta_response.response.reconstructed_wavefunction;
        response.solver = delta_response.solver;
        response.residual_norm = delta_response.residual_norm;
        response.full_grid_equation_residual_norm
            = sternheimer_fd_linear_response_residual_norm(hamiltonian,
                                                            occupied_wavefunctions,
                                                            reference_eigenvalue,
                                                            rhs,
                                                            response.wavefunction,
                                                            frequencies[ifrequency],
                                                            volume_element);
        response.hamiltonian_applications = delta_response.hamiltonian_applications + 1;
        result.hamiltonian_applications += 1;
        response.has_delta_components = true;
        response.delta_components = std::move(delta_response.response);
    }
    return result;
}

std::vector<SternheimerPeriodicLinearResponse> solve_sternheimer_periodic_linear_response_batch(
    const bool use_delta_sternheimer,
    const SternheimerFDHamiltonian& hamiltonian,
    const std::vector<SternheimerFDHamiltonian::Vector>& occupied_wavefunctions,
    const double reference_eigenvalue,
    const SternheimerFDHamiltonian::Matrix& rhs,
    const std::vector<SternheimerDeltaVirtualState>& virtual_states,
    const std::vector<std::vector<SternheimerFDHamiltonian::Complex>>& perturbation_matrix_elements,
    const double omega,
    const double volume_element,
    const SternheimerRPA::SolverOptions& options)
{
    if (rhs.size() != perturbation_matrix_elements.size())
    {
        throw std::invalid_argument(
            "Periodic Sternheimer batch response requires one perturbation-element vector per rhs.");
    }
    std::vector<SternheimerPeriodicLinearResponse> results(rhs.size());
    if (!use_delta_sternheimer)
    {
        for (std::size_t column = 0; column != rhs.size(); ++column)
        {
            results[column] = solve_sternheimer_periodic_linear_response(false,
                                                                          hamiltonian,
                                                                          occupied_wavefunctions,
                                                                          reference_eigenvalue,
                                                                          rhs[column],
                                                                          virtual_states,
                                                                          perturbation_matrix_elements[column],
                                                                          omega,
                                                                          volume_element,
                                                                          options);
        }
        return results;
    }

    const auto dot = [volume_element](const SternheimerFDHamiltonian::Vector& lhs,
                                      const SternheimerFDHamiltonian::Vector& rhs_vector) {
        return sternheimer_fd_grid_dot(lhs, rhs_vector, volume_element);
    };
    for (std::size_t column = 0; column != rhs.size(); ++column)
    {
        results[column].projected_rhs = rhs[column];
        SternheimerRPA::project_out_subspace(occupied_wavefunctions,
                                             dot,
                                             results[column].projected_rhs);
    }

    const SternheimerDeltaFixedSubspace fixed_subspace
        = build_delta_sternheimer_fixed_subspace(occupied_wavefunctions, virtual_states);
    std::vector<SternheimerDeltaLinearResponse> responses
        = solve_delta_sternheimer_linear_response_batch(hamiltonian,
                                                         fixed_subspace,
                                                         reference_eigenvalue,
                                                         rhs,
                                                         virtual_states,
                                                         perturbation_matrix_elements,
                                                         omega,
                                                         volume_element,
                                                         options);
    for (std::size_t column = 0; column != rhs.size(); ++column)
    {
        results[column].wavefunction = responses[column].response.reconstructed_wavefunction;
        results[column].solver = responses[column].solver;
        results[column].residual_norm = responses[column].residual_norm;
        results[column].full_grid_equation_residual_norm
            = sternheimer_fd_linear_response_residual_norm(hamiltonian,
                                                            occupied_wavefunctions,
                                                            reference_eigenvalue,
                                                            rhs[column],
                                                            results[column].wavefunction,
                                                            omega,
                                                            volume_element);
        results[column].has_delta_components = true;
        results[column].delta_components = std::move(responses[column].response);
    }
    return results;
}

} // namespace ModuleRI
