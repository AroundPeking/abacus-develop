#include "sternheimer_siab_primitive_galerkin.h"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace module_ri
{
namespace sternheimer_siab
{
namespace
{

constexpr double rydberg_to_hartree = 0.5;

std::vector<std::complex<double>> scale_to_hartree(
    std::vector<std::complex<double>> matrix_ry)
{
    for (std::complex<double>& value: matrix_ry)
    {
        value *= rydberg_to_hartree;
    }
    return matrix_ry;
}

std::vector<std::complex<double>> rectangular_block(
    const std::vector<std::complex<double>>& square_matrix,
    const std::size_t dimension,
    const std::size_t row_begin,
    const std::size_t row_count,
    const std::size_t column_begin,
    const std::size_t column_count)
{
    if (dimension == 0 || square_matrix.size() != dimension * dimension
        || row_begin > dimension || row_count > dimension - row_begin
        || column_begin > dimension || column_count > dimension - column_begin)
    {
        throw std::invalid_argument(
            "Sternheimer primitive Galerkin matrix block dimensions are inconsistent.");
    }
    std::vector<std::complex<double>> result(row_count * column_count);
    for (std::size_t row = 0; row != row_count; ++row)
    {
        const std::size_t source_offset
            = (row_begin + row) * dimension + column_begin;
        const std::size_t target_offset = row * column_count;
        std::copy_n(square_matrix.begin() + source_offset,
                    column_count,
                    result.begin() + target_offset);
    }
    return result;
}

} // namespace

PrimitiveGalerkinData build_primitive_galerkin_data(
    const std::vector<PrimitiveBlock>& blocks,
    const std::vector<AuxiliaryChannelMetadata>& auxiliary_channels,
    const std::vector<std::vector<std::complex<double>>>& primitive_basis_functions,
    const std::vector<std::vector<std::complex<double>>>& fixed_ao_basis_functions,
    const std::vector<FixedAOSpinInput>& fixed_ao_spins,
    const std::vector<ModuleRI::SternheimerFDHamiltonian>& hamiltonians_ry,
    const std::vector<std::vector<double>>& potentials_ha,
    const std::vector<double>& frequency_ha,
    const std::vector<double>& frequency_weights_ha,
    const double delta_omega)
{
    if (blocks.empty() || primitive_basis_functions.empty()
        || fixed_ao_basis_functions.empty())
    {
        throw std::invalid_argument(
            "Sternheimer primitive Galerkin assembly requires primitive and fixed-AO bases.");
    }
    if (fixed_ao_spins.empty()
        || hamiltonians_ry.size() != fixed_ao_spins.size())
    {
        throw std::invalid_argument(
            "Sternheimer primitive Galerkin spin Hamiltonian dimensions are inconsistent.");
    }
    if (auxiliary_channels.empty()
        || auxiliary_channels.size() != potentials_ha.size()
        || frequency_ha.empty()
        || frequency_ha.size() != frequency_weights_ha.size())
    {
        throw std::invalid_argument(
            "Sternheimer primitive Galerkin channel or frequency dimensions are inconsistent.");
    }

    const std::size_t n_primitive = primitive_basis_functions.size();
    const std::size_t n_fixed_ao = fixed_ao_basis_functions.size();
    const std::size_t combined_dimension = n_primitive + n_fixed_ao;
    std::size_t covered_primitives = 0;
    for (const PrimitiveBlock& block: blocks)
    {
        if (block.offset != static_cast<int>(covered_primitives)
            || block.n_primitive <= 0)
        {
            throw std::invalid_argument(
                "Sternheimer primitive Galerkin blocks must be contiguous.");
        }
        covered_primitives += static_cast<std::size_t>(block.n_primitive);
    }
    if (covered_primitives != n_primitive)
    {
        throw std::invalid_argument(
            "Sternheimer primitive Galerkin blocks do not cover the primitive basis.");
    }

    PrimitiveGalerkinData result;
    result.n_primitive = static_cast<int>(n_primitive);
    result.n_fixed_ao = static_cast<int>(n_fixed_ao);
    result.blocks = blocks;
    result.auxiliary_channels = auxiliary_channels;
    result.frequency_ha = frequency_ha;
    result.frequency_weights_ha = frequency_weights_ha;
    result.overlap_s = overlap_s(primitive_basis_functions, delta_omega);
    result.fixed_ao_grid_overlap
        = overlap_s(fixed_ao_basis_functions, delta_omega);

    std::vector<std::vector<std::complex<double>>> combined_basis_functions;
    combined_basis_functions.reserve(combined_dimension);
    combined_basis_functions.insert(combined_basis_functions.end(),
                                    primitive_basis_functions.begin(),
                                    primitive_basis_functions.end());
    combined_basis_functions.insert(combined_basis_functions.end(),
                                    fixed_ao_basis_functions.begin(),
                                    fixed_ao_basis_functions.end());

    const std::vector<std::vector<std::complex<double>>>
        combined_perturbations_ha
        = perturbation_matrices(combined_basis_functions,
                                potentials_ha,
                                delta_omega);
    result.perturbations_ha.reserve(combined_perturbations_ha.size());
    result.primitive_ao_perturbations_ha.reserve(
        combined_perturbations_ha.size());
    for (const std::vector<std::complex<double>>& matrix:
         combined_perturbations_ha)
    {
        result.perturbations_ha.push_back(rectangular_block(matrix,
                                                            combined_dimension,
                                                            0,
                                                            n_primitive,
                                                            0,
                                                            n_primitive));
        result.primitive_ao_perturbations_ha.push_back(
            rectangular_block(matrix,
                              combined_dimension,
                              0,
                              n_primitive,
                              n_primitive,
                              n_fixed_ao));
    }

    result.primitive_ao_overlap.assign(
        n_primitive * n_fixed_ao, std::complex<double>(0.0, 0.0));
    for (std::size_t fixed_ao = 0; fixed_ao != n_fixed_ao; ++fixed_ao)
    {
        const std::vector<std::complex<double>> ao_primitive_overlap
            = overlap_q(fixed_ao_basis_functions[fixed_ao],
                        primitive_basis_functions,
                        delta_omega);
        for (std::size_t primitive = 0; primitive != n_primitive; ++primitive)
        {
            result.primitive_ao_overlap[primitive * n_fixed_ao + fixed_ao]
                = std::conj(ao_primitive_overlap[primitive]);
        }
    }

    result.spins.reserve(fixed_ao_spins.size());
    for (std::size_t spin = 0; spin != fixed_ao_spins.size(); ++spin)
    {
        const FixedAOSpinInput& input = fixed_ao_spins[spin];
        if (input.spin_index != static_cast<int>(spin)
            || input.occupations.size() != n_fixed_ao)
        {
            throw std::invalid_argument(
                "Sternheimer primitive Galerkin fixed-AO occupations are incomplete.");
        }
        PrimitiveGalerkinSpinData output;
        output.spin_index = input.spin_index;
        output.fixed_ao_occupations = input.occupations;
        const std::vector<std::complex<double>> combined_hamiltonian_ry
            = hamiltonian_matrix(combined_basis_functions,
                                 hamiltonians_ry[spin],
                                 delta_omega);
        output.hamiltonian_ha = scale_to_hartree(rectangular_block(
            combined_hamiltonian_ry,
            combined_dimension,
            0,
            n_primitive,
            0,
            n_primitive));
        output.fixed_ao_grid_hamiltonian_ha
            = scale_to_hartree(rectangular_block(combined_hamiltonian_ry,
                                                  combined_dimension,
                                                  n_primitive,
                                                  n_fixed_ao,
                                                  n_primitive,
                                                  n_fixed_ao));
        output.primitive_ao_hamiltonian_ha
            = scale_to_hartree(rectangular_block(combined_hamiltonian_ry,
                                                  combined_dimension,
                                                  0,
                                                  n_primitive,
                                                  n_primitive,
                                                  n_fixed_ao));
        result.spins.push_back(std::move(output));
    }
    return result;
}

} // namespace sternheimer_siab
} // namespace module_ri
