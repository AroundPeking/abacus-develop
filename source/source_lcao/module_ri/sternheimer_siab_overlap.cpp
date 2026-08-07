#include "sternheimer_siab_overlap.h"

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

    std::vector<Complex> result;
    result.reserve(primitives.size());
    for (const auto& primitive : primitives)
    {
        result.push_back(dot(y, primitive, delta_omega));
    }
    return result;
}

std::vector<Complex> overlap_s(const PrimitiveGrid& primitives, const double delta_omega)
{
    validate_delta_omega(delta_omega);
    validate_primitives(primitives);

    const std::size_t nprimitive = primitives.size();
    std::vector<Complex> result(nprimitive * nprimitive);
    for (std::size_t i = 0; i != nprimitive; ++i)
    {
        for (std::size_t j = i; j != nprimitive; ++j)
        {
            const Complex value = dot(primitives[i], primitives[j], delta_omega);
            result[i * nprimitive + j] = value;
            if (i != j)
            {
                result[j * nprimitive + i] = std::conj(value);
            }
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

    const std::size_t n_basis = basis_functions.size();
    std::vector<std::vector<Complex>> result(potentials_ha.size(),
                                             std::vector<Complex>(n_basis * n_basis, Complex(0.0, 0.0)));
    for (std::size_t channel = 0; channel != potentials_ha.size(); ++channel)
    {
        for (std::size_t row = 0; row != n_basis; ++row)
        {
            for (std::size_t column = row; column != n_basis; ++column)
            {
                Complex value(0.0, 0.0);
                for (std::size_t grid = 0; grid != grid_size; ++grid)
                {
                    value += std::conj(basis_functions[row][grid]) * potentials_ha[channel][grid]
                             * basis_functions[column][grid];
                }
                value *= delta_omega;
                result[channel][row * n_basis + column] = value;
                if (row != column)
                {
                    result[channel][column * n_basis + row] = std::conj(value);
                }
            }
        }
    }
    return result;
}

} // namespace sternheimer_siab
} // namespace module_ri
