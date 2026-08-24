#include "source_lcao/module_ri/sternheimer_coulomb_whitening_complex.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

using Complex = std::complex<double>;

std::vector<Complex> right_multiply(const std::vector<Complex>& left,
                                    const int rows,
                                    const int inner,
                                    const std::vector<Complex>& right,
                                    const int columns)
{
    std::vector<Complex> result(static_cast<std::size_t>(rows * columns), Complex(0.0, 0.0));
    for (int row = 0; row != rows; ++row)
    {
        for (int column = 0; column != columns; ++column)
        {
            for (int index = 0; index != inner; ++index)
            {
                result[static_cast<std::size_t>(row * columns + column)]
                    += left[static_cast<std::size_t>(row * inner + index)]
                       * right[static_cast<std::size_t>(index * columns + column)];
            }
        }
    }
    return result;
}

std::vector<Complex> left_adjoint_multiply(const std::vector<Complex>& left,
                                           const int rows,
                                           const int inner,
                                           const std::vector<Complex>& right,
                                           const int columns)
{
    std::vector<Complex> result(static_cast<std::size_t>(inner * columns), Complex(0.0, 0.0));
    for (int left_column = 0; left_column != inner; ++left_column)
    {
        for (int right_column = 0; right_column != columns; ++right_column)
        {
            for (int row = 0; row != rows; ++row)
            {
                result[static_cast<std::size_t>(left_column * columns + right_column)]
                    += std::conj(left[static_cast<std::size_t>(row * inner + left_column)])
                       * right[static_cast<std::size_t>(row * columns + right_column)];
            }
        }
    }
    return result;
}

} // namespace

TEST(SternheimerComplexCoulombWhitening, ProducesIdentityAndTruncatesNearNullDirection)
{
    const std::vector<Complex> metric = {
        Complex(2.0, 0.0),
        Complex(0.0, 1.0),
        Complex(0.0, 0.0),
        Complex(0.0, -1.0),
        Complex(2.0, 0.0),
        Complex(0.0, 0.0),
        Complex(0.0, 0.0),
        Complex(0.0, 0.0),
        Complex(1.0e-14, 0.0),
    };
    const auto whitening = ModuleRI::make_sternheimer_complex_coulomb_whitening(metric, 3, 1.0e-12);

    EXPECT_EQ(whitening.raw_dimension, 3);
    EXPECT_EQ(whitening.retained_rank, 2);
    EXPECT_EQ(whitening.discarded_rank, 1);
    EXPECT_LT(whitening.max_orthonormality_error, 1.0e-12);

    const auto metric_times_transform = right_multiply(metric, 3, 3, whitening.transform, whitening.retained_rank);
    const auto identity = left_adjoint_multiply(whitening.transform,
                                                3,
                                                whitening.retained_rank,
                                                metric_times_transform,
                                                whitening.retained_rank);
    for (int row = 0; row != whitening.retained_rank; ++row)
    {
        for (int column = 0; column != whitening.retained_rank; ++column)
        {
            const Complex expected = row == column ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
            EXPECT_NEAR(std::abs(identity[static_cast<std::size_t>(row * whitening.retained_rank + column)] - expected),
                        0.0,
                        1.0e-12);
        }
    }
}

TEST(SternheimerComplexCoulombWhitening, DropsNumericallyUnstableRetainedDirections)
{
    const std::vector<Complex> metric = {
        Complex(2.89968841035329383e-01, 0.00000000000000000e+00),
        Complex(5.13810984246140801e-02, -3.56548586400109113e-01),
        Complex(2.55285078010785038e-01, -1.04622708599385730e-01),
        Complex(5.13810984246140801e-02, 3.56548586400109113e-01),
        Complex(4.47525361902515573e-01, 0.00000000000000000e+00),
        Complex(1.73879952906985713e-01, 2.95372017819879651e-01),
        Complex(2.55285078010785038e-01, 1.04622708599385730e-01),
        Complex(1.73879952906985713e-01, -2.95372017819879651e-01),
        Complex(2.62516285260637083e-01, 0.00000000000000000e+00),
    };

    const auto whitening = ModuleRI::make_sternheimer_complex_coulomb_whitening(metric, 3, 1.0e-10);

    EXPECT_EQ(whitening.raw_dimension, 3);
    EXPECT_EQ(whitening.retained_rank, 2);
    EXPECT_EQ(whitening.discarded_rank, 1);
    EXPECT_GT(whitening.smallest_retained_eigenvalue / whitening.largest_eigenvalue, 1.0e-6);
    EXPECT_LE(whitening.max_orthonormality_error, 1.0e-8);
}

TEST(SternheimerComplexCoulombWhitening, RejectsNonHermitianAndMateriallyNegativeMetrics)
{
    const std::vector<Complex> non_hermitian = {
        Complex(2.0, 0.0),
        Complex(0.0, 1.0),
        Complex(0.0, 1.0),
        Complex(2.0, 0.0),
    };
    EXPECT_THROW(ModuleRI::make_sternheimer_complex_coulomb_whitening(non_hermitian, 2, 1.0e-12),
                 std::invalid_argument);

    const std::vector<Complex> negative = {
        Complex(1.0, 0.0),
        Complex(0.0, 0.0),
        Complex(0.0, 0.0),
        Complex(-1.0e-3, 0.0),
    };
    EXPECT_THROW(ModuleRI::make_sternheimer_complex_coulomb_whitening(negative, 2, 1.0e-12), std::runtime_error);
}

TEST(SternheimerComplexCoulombWhitening, ValidatesDimensionsThresholdAndFiniteValues)
{
    const std::vector<Complex> identity = {
        Complex(1.0, 0.0),
        Complex(0.0, 0.0),
        Complex(0.0, 0.0),
        Complex(1.0, 0.0),
    };
    EXPECT_THROW(ModuleRI::make_sternheimer_complex_coulomb_whitening(identity, 0, 1.0e-12), std::invalid_argument);
    EXPECT_THROW(ModuleRI::make_sternheimer_complex_coulomb_whitening(identity, 3, 1.0e-12), std::invalid_argument);
    EXPECT_THROW(ModuleRI::make_sternheimer_complex_coulomb_whitening(identity, 2, 0.0), std::invalid_argument);
    EXPECT_THROW(ModuleRI::make_sternheimer_complex_coulomb_whitening(identity, 2, 1.0), std::invalid_argument);

    std::vector<Complex> non_finite = identity;
    non_finite[0] = Complex(std::numeric_limits<double>::quiet_NaN(), 0.0);
    EXPECT_THROW(ModuleRI::make_sternheimer_complex_coulomb_whitening(non_finite, 2, 1.0e-12), std::invalid_argument);
}
