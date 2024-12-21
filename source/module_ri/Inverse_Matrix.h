//=======================
// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#pragma once

#include "ABFs_Construct-PCA.h"

#include <RI/global/Tensor.h>
#include <vector>

template <typename Tdata>
class Inverse_Matrix
{
  public:
    enum class Method
    {
        potrf,
        syev
    };
    void cal_inverse(const Method& method);

    void input(const RI::Tensor<Tdata>& m);
    void input(const std::vector<std::vector<RI::Tensor<Tdata>>>& ms);
    RI::Tensor<Tdata> output() const;
    std::vector<std::vector<RI::Tensor<Tdata>>> output(const std::vector<size_t>& n0,
                                                       const std::vector<size_t>& n1) const;

  private:
    void using_potrf();
    void using_syev(const double& threshold_condition_number);
    void copy_down_triangle();
    RI::Tensor<Tdata> A;
};

template <>
void Inverse_Matrix<double>::using_syev(const double& threshold_condition_number);

template <>
void Inverse_Matrix<std::complex<double>>::using_syev(const double& threshold_condition_number);

#include "Inverse_Matrix.hpp"