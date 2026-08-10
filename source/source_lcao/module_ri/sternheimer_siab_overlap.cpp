#include "sternheimer_siab_overlap.h"

#include "source_base/module_container/base/third_party/blas.h"
#include <algorithm>
#include "source_base/module_external/blas_connector.h"
#include <cmath>
#include <limits>
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
constexpr std::size_t hamiltonian_batch_target_bytes
    = static_cast<std::size_t>(4) * 1024 * 1024 * 1024;

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

Complex dot(const GridVector& left, const GridVector& right, const double delta_omega)
{
    Complex result(0.0, 0.0);
    for (std::size_t ir = 0; ir != left.size(); ++ir)
    {
        result += std::conj(left[ir]) * right[ir];
    }
    return result * delta_omega;
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
    const double delta_omega,
    const int column_batch_size)
{
    validate_delta_omega(delta_omega);
    const std::size_t grid_size = validate_primitives(basis_functions);
    if (grid_size != static_cast<std::size_t>(hamiltonian.grid().size()))
    {
        throw std::invalid_argument(
            "Sternheimer SIAB Hamiltonian and basis grid sizes differ.");
    }
    if (column_batch_size <= 0)
    {
        throw std::invalid_argument(
            "Sternheimer SIAB Hamiltonian column batch size must be positive.");
    }

    const std::size_t n_basis = basis_functions.size();
    const std::size_t bytes_per_column
        = std::max<std::size_t>(1, grid_size * sizeof(Complex));
    const std::size_t memory_limited_batch_size
        = std::max<std::size_t>(1, hamiltonian_batch_target_bytes / bytes_per_column);
    const std::size_t effective_batch_size
        = std::min({n_basis,
                    static_cast<std::size_t>(column_batch_size),
                    memory_limited_batch_size});
    std::vector<Complex> result(n_basis * n_basis, Complex(0.0, 0.0));
    for (std::size_t first_column = 0; first_column < n_basis;
         first_column += effective_batch_size)
    {
        const std::size_t batch_size
            = std::min(effective_batch_size, n_basis - first_column);
        PrimitiveGrid h_basis(batch_size);
#pragma omp parallel for schedule(dynamic)
        for (int local_column = 0;
             local_column < static_cast<int>(batch_size);
             ++local_column)
        {
            hamiltonian.apply(
                basis_functions[first_column
                                + static_cast<std::size_t>(local_column)],
                h_basis[static_cast<std::size_t>(local_column)]);
        }
        const std::vector<Complex> batch_matrix
            = row_major_matrix_elements(basis_functions,
                                        h_basis,
                                        delta_omega);
        for (std::size_t row = 0; row != n_basis; ++row)
        {
            std::copy_n(batch_matrix.begin() + row * batch_size,
                        batch_size,
                        result.begin() + row * n_basis + first_column);
        }
    }
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

std::vector<Complex> overlap_q_reciprocal(const GridVector& response_coefficients,
                                          const PrimitiveGrid& primitive_coefficients)
{
    if (validate_primitives(primitive_coefficients) != response_coefficients.size())
    {
        throw std::invalid_argument(
            "Sternheimer SIAB response and primitive reciprocal coefficient sizes differ.");
    }
    std::vector<Complex> result;
    result.reserve(primitive_coefficients.size());
    for (const GridVector& primitive: primitive_coefficients)
    {
        result.push_back(dot(response_coefficients, primitive, 1.0));
    }
    return result;
}

std::vector<Complex> overlap_q_reciprocal_contiguous(
    const GridVector& response_coefficients,
    const GridVector& primitive_coefficients,
    const int primitive_count,
    const int reciprocal_count)
{
    if (primitive_count <= 0 || reciprocal_count <= 0
        || response_coefficients.size() != static_cast<std::size_t>(reciprocal_count)
        || static_cast<std::size_t>(primitive_count)
               > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(reciprocal_count)
        || primitive_coefficients.size()
               != static_cast<std::size_t>(primitive_count) * static_cast<std::size_t>(reciprocal_count))
    {
        throw std::invalid_argument("Sternheimer SIAB contiguous reciprocal coefficient dimensions differ.");
    }

    std::vector<Complex> result(static_cast<std::size_t>(primitive_count), Complex(0.0, 0.0));
    // B Y^H gives q_e = sum_G B_eG conj(Y_G), i.e. <Y|B_e>.
    BlasConnector::gemm('N',
                        'C',
                        primitive_count,
                        1,
                        reciprocal_count,
                        Complex(1.0, 0.0),
                        primitive_coefficients.data(),
                        reciprocal_count,
                        response_coefficients.data(),
                        reciprocal_count,
                        Complex(0.0, 0.0),
                        result.data(),
                        1);
    return result;
}

std::vector<Complex> overlap_s_reciprocal(const PrimitiveGrid& primitive_coefficients)
{
    validate_primitives(primitive_coefficients);
    const std::size_t nprimitive = primitive_coefficients.size();
    std::vector<Complex> result(nprimitive * nprimitive);
    for (std::size_t i = 0; i != nprimitive; ++i)
    {
        for (std::size_t j = i; j != nprimitive; ++j)
        {
            const Complex value = dot(primitive_coefficients[i], primitive_coefficients[j], 1.0);
            result[i * nprimitive + j] = value;
            if (i != j)
            {
                result[j * nprimitive + i] = std::conj(value);
            }
        }
    }
    return result;
}

} // namespace sternheimer_siab
} // namespace module_ri
