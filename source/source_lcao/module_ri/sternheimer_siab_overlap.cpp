#include "sternheimer_siab_overlap.h"

#include "source_base/module_container/base/third_party/blas.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace module_ri
{
namespace sternheimer_siab
{
namespace
{

using Complex = std::complex<double>;
using GridVector = std::vector<Complex>;
using PrimitiveGrid = std::vector<GridVector>;

constexpr std::size_t grid_block_size = 32768;

void validate_delta_omega(const double delta_omega)
{
    if (!std::isfinite(delta_omega) || delta_omega <= 0.0)
    {
        throw std::invalid_argument("Sternheimer SIAB overlap requires a positive finite grid volume element.");
    }
}

std::size_t validate_primitives(const PrimitiveGrid& primitives)
{
    if (primitives.empty())
    {
        throw std::invalid_argument("Sternheimer SIAB overlap requires a non-empty primitive basis.");
    }

    const std::size_t grid_size = primitives.front().size();
    for (const auto& primitive : primitives)
    {
        if (primitive.size() != grid_size)
        {
            throw std::invalid_argument("Sternheimer SIAB primitive grids must have equal sizes.");
        }
    }
    return grid_size;
}

std::vector<Complex> row_major_matrix_elements(
    const PrimitiveGrid& left,
    const PrimitiveGrid& right,
    const double delta_omega)
{
    const std::size_t grid_size = validate_primitives(left);
    if (validate_primitives(right) != grid_size)
    {
        throw std::invalid_argument(
            "Sternheimer SIAB matrix-element basis grid sizes differ.");
    }
    const int nleft = static_cast<int>(left.size());
    const int nright = static_cast<int>(right.size());
    std::vector<Complex> column_major(
        static_cast<std::size_t>(nleft) * static_cast<std::size_t>(nright),
        Complex(0.0, 0.0));

    for (std::size_t first = 0; first < grid_size; first += grid_block_size)
    {
        const int block_size = static_cast<int>(
            std::min(grid_block_size, grid_size - first));
        std::vector<Complex> left_block(
            static_cast<std::size_t>(block_size) * left.size());
        std::vector<Complex> right_block(
            static_cast<std::size_t>(block_size) * right.size());
#pragma omp parallel for schedule(static)
        for (int basis = 0; basis < nleft; ++basis)
        {
            for (int grid = 0; grid < block_size; ++grid)
            {
                left_block[static_cast<std::size_t>(basis) * block_size + grid]
                    = left[static_cast<std::size_t>(basis)]
                          [first + static_cast<std::size_t>(grid)];
            }
        }
#pragma omp parallel for schedule(static)
        for (int basis = 0; basis < nright; ++basis)
        {
            for (int grid = 0; grid < block_size; ++grid)
            {
                right_block[static_cast<std::size_t>(basis) * block_size + grid]
                    = right[static_cast<std::size_t>(basis)]
                           [first + static_cast<std::size_t>(grid)];
            }
        }
        container::BlasConnector::gemm('C',
                            'N',
                            nleft,
                            nright,
                            block_size,
                            Complex(delta_omega, 0.0),
                            left_block.data(),
                            block_size,
                            right_block.data(),
                            block_size,
                            Complex(1.0, 0.0),
                            column_major.data(),
                            nleft);
    }

    std::vector<Complex> row_major(column_major.size());
    for (int row = 0; row < nleft; ++row)
    {
        for (int column = 0; column < nright; ++column)
        {
            row_major[static_cast<std::size_t>(row) * nright + column]
                = column_major[static_cast<std::size_t>(row)
                               + static_cast<std::size_t>(column) * nleft];
        }
    }
    return row_major;
}

} // namespace

double norm(const GridVector& y, const double delta_omega)
{
    validate_delta_omega(delta_omega);

    double result = 0.0;
    for (const Complex& value : y)
    {
        result += std::norm(value);
    }
    return result * delta_omega;
}

std::vector<Complex> overlap_q(const GridVector& y,
                               const PrimitiveGrid& primitives,
                               const double delta_omega)
{
    validate_delta_omega(delta_omega);
    if (validate_primitives(primitives) != y.size())
    {
        throw std::invalid_argument("Sternheimer SIAB reference and primitive grid sizes differ.");
    }

    return row_major_matrix_elements({y}, primitives, delta_omega);
}

std::vector<Complex> overlap_s(const PrimitiveGrid& primitives, const double delta_omega)
{
    validate_delta_omega(delta_omega);
    validate_primitives(primitives);

    const std::size_t nprimitive = primitives.size();
    std::vector<Complex> result
        = row_major_matrix_elements(primitives, primitives, delta_omega);
    for (std::size_t row = 0; row != nprimitive; ++row)
    {
        result[row * nprimitive + row]
            = Complex(result[row * nprimitive + row].real(), 0.0);
        for (std::size_t column = row + 1; column != nprimitive; ++column)
        {
            const Complex value
                = 0.5 * (result[row * nprimitive + column]
                         + std::conj(result[column * nprimitive + row]));
            result[row * nprimitive + column] = value;
            result[column * nprimitive + row] = std::conj(value);
        }
    }
    return result;
}

std::vector<std::vector<Complex>> perturbation_matrices(const PrimitiveGrid& basis_functions,
                                                        const std::vector<std::vector<double>>& potentials_ha,
                                                        const double delta_omega)
{
    validate_delta_omega(delta_omega);
    const std::size_t grid_size = validate_primitives(basis_functions);
    if (potentials_ha.empty())
    {
        throw std::invalid_argument("Sternheimer SIAB perturbation matrices require at least one potential.");
    }
    for (const std::vector<double>& potential: potentials_ha)
    {
        if (potential.size() != grid_size)
        {
            throw std::invalid_argument("Sternheimer SIAB potential and basis grid sizes differ.");
        }
        if (!std::all_of(potential.begin(), potential.end(), [](const double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument("Sternheimer SIAB perturbation potential contains a non-finite value.");
        }
    }

    const int n_basis = static_cast<int>(basis_functions.size());
    const std::size_t matrix_size
        = static_cast<std::size_t>(n_basis) * static_cast<std::size_t>(n_basis);
    std::vector<std::vector<Complex>> column_major(
        potentials_ha.size(),
        std::vector<Complex>(matrix_size, Complex(0.0, 0.0)));
    for (std::size_t first = 0; first < grid_size; first += grid_block_size)
    {
        const int block_size = static_cast<int>(
            std::min(grid_block_size, grid_size - first));
        std::vector<Complex> basis_block(
            static_cast<std::size_t>(block_size)
            * static_cast<std::size_t>(n_basis));
        std::vector<Complex> weighted_block(basis_block.size());
#pragma omp parallel for schedule(static)
        for (int basis = 0; basis < n_basis; ++basis)
        {
            for (int grid = 0; grid < block_size; ++grid)
            {
                basis_block[static_cast<std::size_t>(basis) * block_size + grid]
                    = basis_functions[static_cast<std::size_t>(basis)]
                                     [first + static_cast<std::size_t>(grid)];
            }
        }
        for (std::size_t channel = 0; channel != potentials_ha.size(); ++channel)
        {
#pragma omp parallel for schedule(static)
            for (int basis = 0; basis < n_basis; ++basis)
            {
                for (int grid = 0; grid < block_size; ++grid)
                {
                    const std::size_t index
                        = static_cast<std::size_t>(basis) * block_size + grid;
                    weighted_block[index]
                        = basis_block[index]
                          * potentials_ha[channel]
                                         [first + static_cast<std::size_t>(grid)];
                }
            }
            container::BlasConnector::gemm('C',
                                'N',
                                n_basis,
                                n_basis,
                                block_size,
                                Complex(delta_omega, 0.0),
                                basis_block.data(),
                                block_size,
                                weighted_block.data(),
                                block_size,
                                Complex(1.0, 0.0),
                                column_major[channel].data(),
                                n_basis);
        }
    }

    std::vector<std::vector<Complex>> result(
        potentials_ha.size(), std::vector<Complex>(matrix_size));
    for (std::size_t channel = 0; channel != potentials_ha.size(); ++channel)
    {
        for (int row = 0; row < n_basis; ++row)
        {
            result[channel][static_cast<std::size_t>(row) * n_basis + row]
                = Complex(column_major[channel]
                              [static_cast<std::size_t>(row) * (n_basis + 1)]
                                  .real(),
                          0.0);
            for (int column = row + 1; column < n_basis; ++column)
            {
                const Complex value
                    = 0.5
                      * (column_major[channel]
                                          [static_cast<std::size_t>(row)
                                           + static_cast<std::size_t>(column)
                                                 * n_basis]
                         + std::conj(column_major[channel]
                                                [static_cast<std::size_t>(column)
                                                 + static_cast<std::size_t>(row)
                                                       * n_basis]));
                result[channel][static_cast<std::size_t>(row) * n_basis + column]
                    = value;
                result[channel][static_cast<std::size_t>(column) * n_basis + row]
                    = std::conj(value);
            }
        }
    }
    return result;
}

std::vector<Complex> hamiltonian_matrix(
    const PrimitiveGrid& basis_functions,
    const ModuleRI::SternheimerFDHamiltonian& hamiltonian,
    const double delta_omega)
{
    validate_delta_omega(delta_omega);
    const std::size_t grid_size = validate_primitives(basis_functions);
    if (grid_size != static_cast<std::size_t>(hamiltonian.grid().size()))
    {
        throw std::invalid_argument(
            "Sternheimer SIAB Hamiltonian and basis grid sizes differ.");
    }

    const std::size_t n_basis = basis_functions.size();
    PrimitiveGrid h_basis(n_basis);
#pragma omp parallel for schedule(dynamic)
    for (int column = 0; column < static_cast<int>(n_basis); ++column)
    {
        hamiltonian.apply(basis_functions[static_cast<std::size_t>(column)],
                          h_basis[static_cast<std::size_t>(column)]);
    }
    std::vector<Complex> result
        = row_major_matrix_elements(basis_functions, h_basis, delta_omega);
    for (std::size_t row = 0; row != n_basis; ++row)
    {
        result[row * n_basis + row]
            = Complex(result[row * n_basis + row].real(), 0.0);
        for (std::size_t column = row + 1; column != n_basis; ++column)
        {
            const Complex value
                = 0.5 * (result[row * n_basis + column]
                         + std::conj(result[column * n_basis + row]));
            result[row * n_basis + column] = value;
            result[column * n_basis + row] = std::conj(value);
        }
    }
    return result;
}

} // namespace sternheimer_siab
} // namespace module_ri
