#include "gtest/gtest.h"
#include "module_relax/relax_old/bfgsdata.h"  // Include the original code implementation file

std::vector<double> calc_grad_test(std::vector<double> _pos){
    // y = sum_i a_i * (x_i - b_i)^2
    // grad_y_i = 2 * a_i * (x_i - b_i)
    std::vector<double> a = {1.0, 0.5, 2.0};
    std::vector<double> b = {2.0, 1.0, 4.0};

    std::vector<double> _grad(3, 0.0);
    for (int i = 0; i < 3; ++i){
        _grad[i] = 2.0 * a[i] * (_pos[i] - b[i]);
    }
    return _grad;
}

std::vector<double> daxpy_(double scal, int nlen, std::vector<double>& vec1, std::vector<double>& vec2)
{
    std::vector<double> result(nlen);
    for (int i = 0; i < nlen; ++i )
    {
        result[i] =  scal*vec1[i] + vec2[i];
    }
    return result;
}


// Test the VSubV function of BFGS
TEST(BFGS_VSubV_Test, BasicSubtraction) {
    BFGSData bfgsData;
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {0.5, 1.0, 1.5};
    std::vector<double> result = bfgsData.VSubV(a, b);
    std::vector<double> expected = {0.5, 1.0, 1.5};
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], expected[i]);
    }
}

// Test the DotInMAndV1 function of BFGSData
TEST(BFGSData_DotInMAndV1_Test, BasicDotProduct) {
    BFGSData bfgsData;
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::vector<double> vec = {2.0, 3.0};
    std::vector<double> result = bfgsData.DotInMAndV1(matrix, vec);
    std::vector<double> expected = {8.0, 18.0};
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], expected[i]);
    }
}

// Test the DotInMAndV2 function of BFGSData
TEST(BFGSData_DotInMAndV2_Test, BasicDotProduct) {
    BFGSData bfgsData;
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::vector<double> vec = {2.0, 3.0};
    std::vector<double> result = bfgsData.DotInMAndV2(matrix, vec);
    std::vector<double> expected = {11.0, 16.0};
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], expected[i]);
    }
}

// Test the DotInVAndV function of BFGSData
TEST(BFGSData_DotInVAndV_Test, BasicDotProduct) {
    BFGSData bfgsData;
    std::vector<double> vec1 = {1.0, 2.0, 3.0};
    std::vector<double> vec2 = {2.0, 3.0, 4.0};
    double result = bfgsData.DotInVAndV(vec1, vec2);
    double expected = 20.0;
    EXPECT_EQ(result, expected);
}

// Test the OuterVAndV function of BFGSData
TEST(BFGSData_OuterVAndV_Test, BasicOuterProduct) {
    BFGSData bfgsData;
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {3.0, 4.0};
    std::vector<std::vector<double>> result = bfgsData.OuterVAndV(a, b);
    std::vector<std::vector<double>> expected = {
        {3.0, 4.0},
        {6.0, 8.0}
    };
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[0].size(); ++j) {
            EXPECT_EQ(result[i][j], expected[i][j]);
        }
    }
}

// Test the ScaleMatrix function of BFGSData
TEST(BFGSData_ScaleMatrix_Test, BasicScale) {
    BFGSData bfgsData;
    std::vector<std::vector<double>> a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    double b = 2.0;
    std::vector<std::vector<double>> result = bfgsData.ScaleMatrix(a, b);
    std::vector<std::vector<double>> expected = {
        {2.0, 4.0},
        {6.0, 8.0}
    };
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[0].size(); ++j) {
            EXPECT_EQ(result[i][j], expected[i][j]);
        }
    }
}

// Test the MatAdd function of BFGSData
TEST(BFGSData_MatAdd_Test, BasicSubtraction) {
    BFGSData bfgsData;
    std::vector<std::vector<double>> a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::vector<std::vector<double>> b = {
        {0.5, 1.0},
        {1.5, 2.0}
    };
    std::vector<std::vector<double>> result = bfgsData.MatAdd(a, b);
    std::vector<std::vector<double>> expected = {
        {1.5, 3.0},
        {4.5, 6.0}
    };
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[0].size(); ++j) {
            EXPECT_EQ(result[i][j], expected[i][j]);
        }
    }
}

// Test first BFGS Step
TEST(BFGSData_RelaxStep_Test, BFGSFirstStep){
    BFGSData bfgsData;
    std::vector<double> my_pos = {1.0, 2.0, 3.0};
    std::vector<double> my_grad = {0.3, 0.7, 0.5};
    bfgsData.allocate(3);
    bfgsData.set_alpha(2.0);
//    bfgsData.RecPrtMat("initial hessian", bfgsData.H, 3, 3);
    std::vector<std::vector<double>> expectedH = {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}};
    for (size_t i = 0; i < expectedH.size(); ++i) {
        for (size_t j = 0; j < expectedH[0].size(); ++j) {
            EXPECT_EQ(bfgsData.H[i][j], expectedH[i][j]);
        }
    }
    std::vector<double> result = bfgsData.relax_step(my_pos, my_grad);
//    bfgsData.RecPrtVec("result", result, 1, 3);

//    bfgsData.RecPrtMat("updated hessian (should not change)", bfgsData.H, 3, 3);
    for (size_t i = 0; i < expectedH.size(); ++i) {
        for (size_t j = 0; j < expectedH[0].size(); ++j) {
            EXPECT_EQ(bfgsData.H[i][j], expectedH[i][j]);
        }
    }
    std::vector<double> expected = {0.15, 0.35, 0.25};
    for (size_t i = 0; i < result.size(); ++i) {
       EXPECT_EQ(result[i], expected[i]);
    }
}


// Test BFGS based on the function calc_grad_test
TEST(BFGSData_RelaxStep_Test, BFGSIterations){
    BFGSData bfgsData;
    std::vector<double> my_pos = {1.0, 2.0, 3.0};
    bfgsData.allocate(3);
    bfgsData.set_alpha(2.0);

    std::vector<double> my_grad(3);
    std::vector<double> deltapos(3);

//  first iteration
    for (int i_iter = 0; i_iter < 10; ++i_iter){
        my_grad = calc_grad_test(my_pos);
//        bfgsData.RecPrtVec("my_pos", my_pos, 1, 3);
//        bfgsData.RecPrtVec("my_grad", my_grad, 1, 3);
        deltapos = bfgsData.relax_step(my_pos, my_grad);
//        bfgsData.RecPrtMat("hessian", bfgsData.H, 3, 3);
//        bfgsData.RecPrtVec("deltapos", deltapos, 1, 3);
        my_pos = daxpy_(-1.0, 3, deltapos, my_pos);
//        bfgsData.RecPrtVec("my_pos", my_pos, 1, 3);
//        std::cout << std::endl;
    }
    std::vector<double> expected = {2.0, 1.0, 4.0};

    for (int i = 0; i < 3; ++i){
        EXPECT_NEAR(my_pos[i], expected[i], 1e-8);
    }

}


 
