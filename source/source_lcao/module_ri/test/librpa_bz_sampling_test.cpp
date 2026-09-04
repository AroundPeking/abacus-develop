#include "../librpa_bz_sampling.h"

#include "source_base/constants.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

void test_stored_irreducible_points_keep_identity_labels()
{
    constexpr int stored_points = 8;
    require(RpaLriDetail::librpa_stored_coulomb_q_count(stored_points) == stored_points,
            "every stored Coulomb q point must be declared");
    for (int ik = 0; ik != stored_points; ++ik)
    {
        const auto index = RpaLriDetail::librpa_stored_q_index(ik, stored_points);
        require(index.coulomb_irreducible_index == ik + 1,
                "stored Coulomb q labels must be one-based and unique");
        require(index.representative_scf_index == ik + 1,
                "stored SCF representative labels must refer to their own rows");
    }
}

void test_cartesian_scale_uses_actual_lattice_constant()
{
    constexpr double lat0_bohr = 7.993541507167009;
    const double actual = RpaLriDetail::librpa_bz_cartesian_scale(lat0_bohr);
    const double expected = ModuleBase::TWO_PI / lat0_bohr;
    require(std::abs(actual - expected) <= 4.0 * std::numeric_limits<double>::epsilon(),
            "BZ Cartesian coordinates must be written in Bohr^-1 using 2*pi/lat0");

    const double one_angstrom_bohr = 1.0 / ModuleBase::BOHR_TO_A;
    require(RpaLriDetail::librpa_bz_cartesian_scale(one_angstrom_bohr)
                == ModuleBase::TWO_PI * ModuleBase::BOHR_TO_A,
            "the one-Angstrom convention must retain its legacy numerical values");
}

void test_invalid_counts_and_indices_are_rejected()
{
    bool rejected = false;
    try
    {
        (void)RpaLriDetail::librpa_stored_coulomb_q_count(0);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected, "an empty stored q mesh must be rejected");

    for (const int ik: {-1, 8})
    {
        rejected = false;
        try
        {
            (void)RpaLriDetail::librpa_stored_q_index(ik, 8);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        require(rejected, "out-of-range stored q indices must be rejected");
    }
}
} // namespace

int main()
{
    test_stored_irreducible_points_keep_identity_labels();
    test_cartesian_scale_uses_actual_lattice_constant();
    test_invalid_counts_and_indices_are_rejected();
    std::cout << "LibRPA BZ sampling tests passed\n";
    return 0;
}
