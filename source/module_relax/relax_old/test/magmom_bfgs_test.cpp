#include "module_base/matrix3.h"
#include "gtest/gtest.h"
#include "module_relax/relax_old/magmom_bfgs.h"  // Include the original code implementation file
#include "module_hamilt_lcao/module_deltaspin/spin_constrain.h"

// Test the functionality of the Magmom_BFGS class
class MagmomBFGS_OptimizerTest : public ::testing::Test {
protected:
    Magmom_BFGS optimizer;
    void SetUp() override {
        // Initialization operation performed before each test case
        optimizer.initialize(2, 3);
    }
};


// Test the initialize function
//TEST_F(MagmomBFGS_OptimizerTest, InitializeTest) {
//    EXPECT_EQ(optimizer.n_atom, 2);
//    EXPECT_EQ(optimizer.n_moment_component, 3);
//    EXPECT_EQ(optimizer.max_delta_moment, 0.2);
//    EXPECT_EQ(optimizer.convergence_threshold, 1.0e-5);
//    EXPECT_FALSE(optimizer.converged);
//}


// Test the calc_new_magmom function
TEST_F(MagmomBFGS_OptimizerTest, CalcNewMagmomTest) {
    std::vector<std::vector<double>> magmom = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<std::vector<double>> magforce = {{-1.0, -2.0, -3.0}, {-4.0, -5.0, -6.0}};
    std::vector<std::vector<double>> expected_result = {{1.25, 2.5, 3.75}, {5.0, 6.25, 7.5}};  // Assume the expected result
    optimizer.set_max_delta_moment(10.0);
    optimizer.set_alpha(4.0);
    std::vector<std::vector<double>> result = optimizer.calc_new_magmom(magmom, magforce);
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[0].size(); ++j) {
            EXPECT_DOUBLE_EQ(result[i][j], expected_result[i][j]);
        }
    }
}


// Test the calc_moment_module function
TEST_F(MagmomBFGS_OptimizerTest, CalcMomentModuleTest) {
    std::vector<std::vector<double>> mat_moment = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<double> expected_result = {std::sqrt(14.0), std::sqrt(77.0)};
    std::vector<double> result = optimizer.calc_moment_module(mat_moment, 2, 3);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], expected_result[i]);
    }
}


// Test the mat_to_vec function
TEST_F(MagmomBFGS_OptimizerTest, MatToVecTest) {
    std::vector<std::vector<double>> mat = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<double> expected_result = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> result = optimizer.mat_to_vec(mat, 2, 3);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], expected_result[i]);
    }
}


// Test the vec_to_mat function
TEST_F(MagmomBFGS_OptimizerTest, VecToMatTest) {
    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<std::vector<double>> expected_result = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<std::vector<double>> result = optimizer.vec_to_mat(vec, 2, 3);
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[0].size(); ++j) {
            EXPECT_DOUBLE_EQ(result[i][j], expected_result[i][j]);
        }
    }
}


// Test the set_convergence_threshold function
//TEST_F(MagmomBFGS_OptimizerTest, SetConvergenceThresholdTest) {
//    optimizer.set_convergence_threshold(1e-6);
//    EXPECT_DOUBLE_EQ(optimizer.convergence_threshold, 1e-6);
//}


// Test the set_max_delta_moment function
//TEST_F(MagmomBFGS_OptimizerTest, SetMaxDeltaMomentTest) {
//    optimizer.set_max_delta_moment(0.3);
//    EXPECT_DOUBLE_EQ(optimizer.max_delta_moment, 0.3);
//}


// Test the get_convergence_status function
//TEST_F(MagmomBFGS_OptimizerTest, GetConvergenceStatusTest) {
//    optimizer.converged = true;
//    EXPECT_TRUE(optimizer.get_convergence_status());
//    optimizer.converged = false;
//    EXPECT_FALSE(optimizer.get_convergence_status());
//}


// Test the FindAbsMaxVec function
TEST_F(MagmomBFGS_OptimizerTest, FindAbsMaxVecTest) {
    std::vector<double> vec = {-1.0, 2.0, -3.0, 4.0};
    double expected_result = 4.0;
    EXPECT_DOUBLE_EQ(optimizer.FindAbsMaxVec(vec), expected_result);
}


// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
