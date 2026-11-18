#include "source_lcao/module_ri/exx_abfs_ctor_orbs.h"
#include "gtest/gtest.h"

#include <vector>

/************************************************
 *  unit test of Exx_Abfs::Construct_Orbs::get_Lmax
 ***********************************************/

/**
 * - Tested Function
 *   - get_Lmax on a populated channel list
 *   - get_Lmax on a list whose trailing channels were filtered away
 *   - get_Lmax on an all-empty list
 *   - get_Lmax on an empty list of atom types, which used to dereference end()
 *   - the sum that Matrix_Orbs11::init hands to Center2_Orb
 *
 * get_Lmax used to be `max_element(...)->size() - 1`. Two things were wrong with
 * that expression:
 *
 *   1. An empty list of atom types made max_element return end(), and the
 *      dereference of end() is undefined behaviour.
 *   2. size() is unsigned, so an all-empty list computed 0 - 1 = SIZE_MAX and
 *      recovered -1 only through the int conversion.
 *
 * The crash those enabled came from a consumer: Matrix_Orbs11::init sums two
 * get_Lmax results, Center2_Orb::init_Table_Spherical_Bessel adds one more, and
 * Sph_Bessel_Recursive::cal_jlx rejects a negative lmax. Two empty operands gave
 * -2, so the +1 stayed below zero and threw "jlx l<0".
 *
 * The empty operand was an auxiliary basis that never got built: a PCA threshold
 * above 1 makes abfs_same_atom return one empty channel list per atom type, and a
 * STRU with no ABFS_ORBITAL block then leaves it empty. prepare_abfs now reports
 * that as an input error. These tests pin the arithmetic that used to hide it.
 */

TEST(ExxAbfsGetLmax, PopulatedChannels)
{
    // Two atom types; the second carries the highest angular momentum.
    std::vector<std::vector<int>> orb(2);
    orb[0] = {0, 1, 2};
    orb[1] = {0, 1, 2, 3};
    EXPECT_EQ(Exx_Abfs::Construct_Orbs::get_Lmax(orb), 3);
}

TEST(ExxAbfsGetLmax, TrailingChannelsFilteredAway)
{
    // filter_empty_orbs() trims only from the end, so a type whose high channels
    // were emptied keeps a shorter list rather than vanishing entirely.
    std::vector<std::vector<int>> orb(2);
    orb[0] = {0, 1};
    orb[1] = {};
    EXPECT_EQ(Exx_Abfs::Construct_Orbs::get_Lmax(orb), 1);
}

TEST(ExxAbfsGetLmax, AllChannelsEmpty)
{
    // What an unbuilt auxiliary basis looks like. -1 means "no angular momentum
    // present"; the old expression reached the same number through an unsigned
    // wrap, which is why the defect looked harmless until the values were summed.
    std::vector<std::vector<int>> orb(2);
    orb[0] = {};
    orb[1] = {};
    EXPECT_EQ(Exx_Abfs::Construct_Orbs::get_Lmax(orb), -1);
}

TEST(ExxAbfsGetLmax, EmptyListOfTypes)
{
    // max_element on an empty range returns end(); dereferencing it was undefined
    // behaviour. This case must simply report no angular momentum.
    std::vector<std::vector<int>> orb;
    EXPECT_EQ(Exx_Abfs::Construct_Orbs::get_Lmax(orb), -1);
}

TEST(ExxAbfsGetLmax, SumOfTwoEmptyOperandsReachesTheConsumerAsNegative)
{
    // Matrix_Orbs11::init sums the two operands. Both empty gives -2, whose +1 is
    // still below the zero cal_jlx accepts, so Center2_Orb has to skip the table.
    std::vector<std::vector<int>> empty_operand(1);
    empty_operand[0] = {};
    const int lmax_used = Exx_Abfs::Construct_Orbs::get_Lmax(empty_operand)
                          + Exx_Abfs::Construct_Orbs::get_Lmax(empty_operand);
    EXPECT_EQ(lmax_used, -2);
    EXPECT_LT(lmax_used + 1, 0);
}
