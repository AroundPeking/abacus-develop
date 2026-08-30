#include "source_lcao/module_ri/sternheimer_runtime_options.h"

#include <cstdlib>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace
{

constexpr const char* kTestFlag = "ABACUS_STERNHEIMER_TEST_FLAG";
constexpr const char* kBatchWidth = "ABACUS_STERNHEIMER_CHANNEL_BATCH_WIDTH";
constexpr const char* kFrequencyRecycling = "ABACUS_STERNHEIMER_FREQUENCY_RECYCLING";
constexpr const char* kFrequencyGroupSize = "ABACUS_STERNHEIMER_FREQUENCY_RECYCLING_GROUP_SIZE";
constexpr const char* kFrequencyBasisDimension
    = "ABACUS_STERNHEIMER_FREQUENCY_RECYCLING_MAX_BASIS_DIMENSION";

class SternheimerRuntimeOptionsTest : public ::testing::Test
{
  protected:
    void TearDown() override
    {
        unsetenv(kTestFlag);
        unsetenv(kBatchWidth);
        unsetenv(kFrequencyRecycling);
        unsetenv(kFrequencyGroupSize);
        unsetenv(kFrequencyBasisDimension);
    }
};

} // namespace

TEST_F(SternheimerRuntimeOptionsTest, UsesDefaultWhenEnvironmentIsMissing)
{
    unsetenv(kTestFlag);

    EXPECT_TRUE(ModuleRI::sternheimer_environment_flag(kTestFlag, true));
    EXPECT_FALSE(ModuleRI::sternheimer_environment_flag(kTestFlag, false));
}

TEST_F(SternheimerRuntimeOptionsTest, UsesTwoChannelBatchesByDefaultAndOneForRollback)
{
    unsetenv(kBatchWidth);
    EXPECT_EQ(ModuleRI::sternheimer_channel_batch_width(), 2);

    setenv(kBatchWidth, "1", 1);
    EXPECT_EQ(ModuleRI::sternheimer_channel_batch_width(), 1);
    setenv(kBatchWidth, "8", 1);
    EXPECT_EQ(ModuleRI::sternheimer_channel_batch_width(), 8);
}

TEST_F(SternheimerRuntimeOptionsTest, RejectsInvalidChannelBatchWidths)
{
    for (const char* value: {"", "0", "-1", "four", "65"})
    {
        setenv(kBatchWidth, value, 1);
        EXPECT_THROW(ModuleRI::sternheimer_channel_batch_width(), std::invalid_argument) << value;
    }
}

TEST_F(SternheimerRuntimeOptionsTest, ParsesRecognizedBooleanTextCaseInsensitively)
{
    for (const char* value: {"1", "true", "TRUE", "on", "On", "yes", "YES"})
    {
        setenv(kTestFlag, value, 1);
        EXPECT_TRUE(ModuleRI::sternheimer_environment_flag(kTestFlag, false)) << value;
    }
    for (const char* value: {"0", "false", "FALSE", "off", "Off", "no", "NO"})
    {
        setenv(kTestFlag, value, 1);
        EXPECT_FALSE(ModuleRI::sternheimer_environment_flag(kTestFlag, true)) << value;
    }
}

TEST_F(SternheimerRuntimeOptionsTest, RejectsEmptyAndUnknownBooleanText)
{
    for (const char* value: {"", "maybe", "2"})
    {
        setenv(kTestFlag, value, 1);
        EXPECT_THROW(ModuleRI::sternheimer_environment_flag(kTestFlag, true), std::invalid_argument) << value;
    }
}

TEST_F(SternheimerRuntimeOptionsTest, DisablesFrequencyRecyclingByDefault)
{
    const auto options = ModuleRI::sternheimer_frequency_recycling_runtime_options();

    EXPECT_FALSE(options.enabled);
    EXPECT_EQ(options.group_size, 3);
    EXPECT_EQ(options.max_basis_dimension, 48);
}

TEST_F(SternheimerRuntimeOptionsTest, ParsesFrequencyRecyclingExperimentOptions)
{
    setenv(kFrequencyRecycling, "true", 1);
    setenv(kFrequencyGroupSize, "4", 1);
    setenv(kFrequencyBasisDimension, "32", 1);

    const auto options = ModuleRI::sternheimer_frequency_recycling_runtime_options();

    EXPECT_TRUE(options.enabled);
    EXPECT_EQ(options.group_size, 4);
    EXPECT_EQ(options.max_basis_dimension, 32);
}

TEST_F(SternheimerRuntimeOptionsTest, RejectsInvalidFrequencyRecyclingDimensions)
{
    for (const char* value: {"", "0", "-1", "three", "65"})
    {
        setenv(kFrequencyGroupSize, value, 1);
        EXPECT_THROW(ModuleRI::sternheimer_frequency_recycling_runtime_options(),
                     std::invalid_argument)
            << value;
    }
    unsetenv(kFrequencyGroupSize);
    for (const char* value: {"", "0", "-1", "wide", "513"})
    {
        setenv(kFrequencyBasisDimension, value, 1);
        EXPECT_THROW(ModuleRI::sternheimer_frequency_recycling_runtime_options(),
                     std::invalid_argument)
            << value;
    }
}
