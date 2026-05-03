#include "gtest/gtest.h"

#include "source_io/module_parameter/input_parameter.h"
#include "source_lcao/module_ri/Exx_LRI.h"

namespace
{
TEST(ExxLriRotationPolicy, EnablesMomentRotationForRpaByDefault)
{
    Input_para inp;
    inp.rpa = true;
    inp.exx_rotate_abfs = false;
    inp.rpa_rotate_abfs = true;

    EXPECT_TRUE(ExxLriDetail::should_rotate_abfs_for_init(inp));
}

TEST(ExxLriRotationPolicy, DisablesMomentRotationForRpaWhenSwitchIsOff)
{
    Input_para inp;
    inp.rpa = true;
    inp.exx_rotate_abfs = false;
    inp.rpa_rotate_abfs = false;

    EXPECT_FALSE(ExxLriDetail::should_rotate_abfs_for_init(inp));
}

TEST(ExxLriRotationPolicy, PreservesLegacyExxRotateAbfsBehaviorOutsideRpa)
{
    Input_para inp;
    inp.rpa = false;
    inp.exx_rotate_abfs = true;
    inp.rpa_rotate_abfs = false;

    EXPECT_TRUE(ExxLriDetail::should_rotate_abfs_for_init(inp));
}

TEST(ExxLriRotationPolicy, LegacyRotateAbfsStillWinsInsideRpa)
{
    Input_para inp;
    inp.rpa = true;
    inp.exx_rotate_abfs = true;
    inp.rpa_rotate_abfs = false;

    EXPECT_TRUE(ExxLriDetail::should_rotate_abfs_for_init(inp));
}
} // namespace
