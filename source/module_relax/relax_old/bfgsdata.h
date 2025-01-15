#ifndef BFGS_DATA_H
#define BFGS_DATA_H

/**
 * @file bfgs.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <vector>
#include <tuple> 
#include<algorithm>
#include<cmath>
#include<string.h>
#include"module_base/lapack_connector.h"

#include "module_base/matrix.h"
#include "module_base/matrix3.h"



class BFGSData
{
public:
    
    double alpha;//initialize H,diagonal element is alpha
    int size; // dimension of vector
    int nelement; // size^2
    
    std::vector<std::vector<double>> H;
    std::vector<double> grad0;
    std::vector<double> grad;
    std::vector<double> pos0;
    std::vector<double> pos;
    std::vector<double> deltapos;

    /**
     * @brief 
     * 
     * @param _size 
     */
    void allocate(const int _size);//initialize parameters
    void set_alpha(double _alpha);
    std::vector<double> relax_step(const std::vector<double>& _pos, const std::vector<double>& _grad);//
    void CalculateDeltaStep();
    BFGSData();
    ~BFGSData();

    std::vector<double> VSubV(std::vector<double>& a, std::vector<double>& b);
    std::vector<double> DotInMAndV1(std::vector<std::vector<double>>& matrix, std::vector<double>& vec);
    std::vector<double> DotInMAndV2(std::vector<std::vector<double>>& matrix, std::vector<double>& vec);
    std::vector<std::vector<double>> ScaleMatrix(std::vector<std::vector<double>>& a, double& b);
    double DotInVAndV(std::vector<double>& vec1, std::vector<double>& vec2);
    std::vector<std::vector<double>> OuterVAndV(std::vector<double>& a, std::vector<double>& b);
    std::vector<std::vector<double>> MatAdd(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b);
    double FindMaxVec(std::vector<double> vec);
    void RecPrtVec(std::string info, std::vector<double>& vec, int nrow, int ncol);
    void RecPrtMat(std::string info, std::vector<std::vector<double>>& mat, int nrow, int ncol);
private:
    bool first_step;

    void UpdateHessian();

    // matrix method
};

#endif // BFGS_H
