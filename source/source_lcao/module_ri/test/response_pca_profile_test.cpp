#include "../response_pca_profile.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace
{

TEST(ResponsePCAProfile, EmptySpecificationKeepsLegacyMode)
{
    EXPECT_TRUE(ResponsePCA::parse_fixed_nu_profiles("").empty());
}

TEST(ResponsePCAProfile, ParsesOneProfilePerAtomType)
{
    const ResponsePCA::FixedNuProfiles profiles
        = ResponsePCA::parse_fixed_nu_profiles("2,2,1,0,0;1,1");

    ASSERT_EQ(profiles.size(), 2U);
    EXPECT_EQ(profiles[0], (std::vector<std::size_t>{2, 2, 1, 0, 0}));
    EXPECT_EQ(profiles[1], (std::vector<std::size_t>{1, 1}));
}

TEST(ResponsePCAProfile, RejectsMalformedProfiles)
{
    EXPECT_THROW(ResponsePCA::parse_fixed_nu_profiles("2,,1"), std::invalid_argument);
    EXPECT_THROW(ResponsePCA::parse_fixed_nu_profiles("2,-1,1"), std::invalid_argument);
    EXPECT_THROW(ResponsePCA::parse_fixed_nu_profiles("2,x,1"), std::invalid_argument);
    EXPECT_THROW(ResponsePCA::parse_fixed_nu_profiles("2,1;"), std::invalid_argument);
}

TEST(ResponsePCAProfile, ValidatesTypeCountAndAvailableRadialFunctions)
{
    const ResponsePCA::FixedNuProfiles profiles
        = ResponsePCA::parse_fixed_nu_profiles("2,2,1,0,0;1,1");
    const std::vector<std::vector<std::size_t>> available{{3, 3, 2, 1, 3}, {2, 1}};
    EXPECT_NO_THROW(ResponsePCA::validate_fixed_nu_profiles(profiles, available));

    EXPECT_THROW(ResponsePCA::validate_fixed_nu_profiles(profiles, {{3, 3, 2, 1, 3}}),
                 std::invalid_argument);
    EXPECT_THROW(ResponsePCA::validate_fixed_nu_profiles(
                     ResponsePCA::parse_fixed_nu_profiles("4,2,1,0,0;1,1"), available),
                 std::invalid_argument);
    EXPECT_THROW(ResponsePCA::validate_fixed_nu_profiles(
                     ResponsePCA::parse_fixed_nu_profiles("0,0,0,0,0;1,1"), available),
                 std::invalid_argument);
}

TEST(ResponsePCAProfile, KeepsOnlyProductsTouchingTheFixedPrefix)
{
    const std::vector<std::size_t> fixed_nu{2, 1, 0};

    EXPECT_TRUE(ResponsePCA::keep_radial_product({}, 0, 2, 2, 1));
    EXPECT_TRUE(ResponsePCA::keep_radial_product(fixed_nu, 0, 0, 2, 1));
    EXPECT_TRUE(ResponsePCA::keep_radial_product(fixed_nu, 2, 1, 1, 0));
    EXPECT_FALSE(ResponsePCA::keep_radial_product(fixed_nu, 0, 2, 2, 1));
}

TEST(ResponsePCAProfile, ExpandsRadialPrefixToAllMagneticComponents)
{
    const std::vector<std::size_t> fixed_nu{1, 1, 0};
    const std::vector<std::size_t> available_nu{2, 1, 1};
    const std::vector<bool> mask = ResponsePCA::make_fixed_ao_mask(fixed_nu, available_nu);

    ASSERT_EQ(mask.size(), 10U);
    EXPECT_EQ(mask, (std::vector<bool>{true, false, true, true, true, false, false, false, false, false}));
    EXPECT_EQ(ResponsePCA::count_kept_ordered_pairs(mask), 64U);
}

} // namespace
