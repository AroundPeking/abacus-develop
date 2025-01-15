#ifndef MAGMOM_BFGS_H
#define MAGMOM_BFGS_H

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
#include <algorithm>
#include <cmath>
#include <string.h>
#include "module_base/lapack_connector.h"
#include "bfgsdata.h"
#include "module_base/matrix.h"
#include "module_base/matrix3.h"
#include "module_hamilt_lcao/module_deltaspin/spin_constrain.h"



class Magmom_BFGS : public BFGSData
{

private:
    int n_atom;
    int n_moment_component;
    double max_delta_moment;
    double convergence_threshold;
    bool converged;

    std::vector<double> vec_delta_moment;
    std::vector<double> vec_mag_moment;
    std::vector<double> vec_mag_force;

public:
//    std::vector<std::vector<double>> mag_moment;
//    std::vector<std::vector<double>> mag_force;
    std::vector<std::vector<double>> delta_moment;
    std::vector<std::vector<double>> new_moment;
    std::vector<double>  module_moment;
    std::vector<double>  module_force;

    Magmom_BFGS();
    ~Magmom_BFGS();

    std::vector<std::vector<double>> calc_new_magmom(std::vector<std::vector<double>> _magmom, std::vector<std::vector<double>> _magforce);
    std::vector<double> mat_to_vec(std::vector<std::vector<double>> mat, int nrow, int ncol);
    std::vector<std::vector<double>> vec_to_mat(std::vector<double> vec, int nrow, int ncol);

    std::vector<double> calc_moment_module(std::vector<std::vector<double>> mat_moment, int nrow, int ncol);
    void set_max_delta_moment(double _max_delta_moment);
    void set_convergence_threshold(double _convergence_threshold);
    bool get_convergence_status();
    bool bfgs_wrapper();
    std::vector<std::vector<double>> matvec3_to_mat(std::vector<ModuleBase::Vector3<double>> matvec3, int nrow);
    std::vector<ModuleBase::Vector3<double>> mat_to_matvec3(std::vector<std::vector<double>> mat, int nrow);
    void initialize(const int _n_atom, const int _n_moment_component);
    double FindAbsMaxVec(std::vector<double> vec);
    void constrain_magforce(std::vector<ModuleBase::Vector3<double>>& magforce, std::vector<ModuleBase::Vector3<int>> magconstrain);
};
#endif // MAGFORCE_BFGS_H
