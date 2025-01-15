#include "bfgsdata.h"
#include "module_base/matrix3.h"

BFGSData::BFGSData(){}
BFGSData::~BFGSData(){}

//! initialize H0、H、pos0、grad0、grad
void BFGSData::allocate(const int _size) 
{
    alpha=1.0;
    size=_size;
    nelement = size * size;
    first_step =true;
    
    H = std::vector<std::vector<double>>(size, std::vector<double>(size, 0.0));
    
    for (int i = 0; i < size; ++i) 
    {
        H[i][i] = alpha;  
    }
    
    pos = std::vector<double> (size, 0.0); 
    pos0 = std::vector<double>(size, 0.0);
    deltapos = std::vector<double>(size, 0.0);
    grad = std::vector<double>(size, 0.0);
    grad0 = std::vector<double>(size, 0.0);
}

void BFGSData::set_alpha(double _alpha)
{
    alpha = _alpha;
    for (int i = 0; i < size; ++i) 
    {
        H[i][i] = alpha;  
    }
}


std::vector<double> BFGSData::relax_step(const std::vector<double>& _pos, const std::vector<double>& _grad) 
{
    grad = _grad;
    pos = _pos;   
 
    this->CalculateDeltaStep();
    return deltapos; 
}

void BFGSData::CalculateDeltaStep()
{
    this->UpdateHessian();
    
    //! call dysev
    std::vector<double> eigenvalue(size);
    std::vector<double> work(nelement);
    std::vector<double> grad_copy = grad;
    int lwork=nelement;
    int info=0;
    std::vector<double> H_flat;
    
    for(const auto& row : H)
    {
        H_flat.insert(H_flat.end(), row.begin(), row.end());
    }
    
    int* size_ptr=&size;

    dsyev_("V","U",size_ptr,H_flat.data(),size_ptr,eigenvalue.data(),work.data(),&lwork,&info);

    std::vector<std::vector<double>> V(size, std::vector<double>(size, 0.0));

    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            V[j][i] = H_flat[size*i + j];
        }
    }

    std::vector<double> rotated_grad=this->DotInMAndV2(V, grad_copy);

    for(int i = 0; i < rotated_grad.size(); i++)
    {
        rotated_grad[i]/=std::abs(eigenvalue[i]);    
    }

    deltapos = this->DotInMAndV1(V, rotated_grad);
    pos0 = pos;
    grad0 = grad;
}

void BFGSData::UpdateHessian()
{
//    std::cout << "first step = " << first_step << std::endl;
    if(first_step)
    {
        first_step=false;
        return;
    }

    std::vector<double> dgrad = this->VSubV(grad, grad0);
    std::vector<double> dpos = this->VSubV(pos, pos0);

//    std::cout << "printing intermediate BFGS results" << std::endl;
//    this->RecPrtVec("dgrad", dgrad, 1, size);
//    this->RecPrtVec("dpos", dpos, 1, size);
    double a = 1.0/this->DotInVAndV(dpos, dgrad);
    std::vector<double> dg = this->DotInMAndV1(H, dpos);
//    this->RecPrtMat("hessian", H, size, size);
//    this->RecPrtVec("dpos", dpos, size, 1);
//    this->RecPrtVec("hessian*dpos", dg, size, 1);
    double b = -1.0/this->DotInVAndV(dpos, dg);
//    std::cout << "denominator 1: " << a << std::endl;
//    std::cout << "denominator 2: " << b << std::endl;
//    std::cout << "1/denominator 2: " << this->DotInVAndV(dpos, dg) << std::endl;
    auto term1=this->OuterVAndV(dgrad, dgrad);
//    this->RecPrtMat("term1", term1, size, size);
    auto term2=this->OuterVAndV(dg, dg);
//    this->RecPrtMat("term2", term2, size, size);

    auto term3=this->ScaleMatrix(term1, a);
    auto term4=this->ScaleMatrix(term2, b);
    H = this->MatAdd(H, term3);
    H = this->MatAdd(H, term4);
}

std::vector<double> BFGSData::VSubV(std::vector<double>& a, std::vector<double>& b) 
{
    std::vector<double> result = std::vector<double>(a.size(), 0.0);
    for(int i = 0; i < a.size(); i++)
    {
        result[i] = a[i] - b[i];
    }
    return result;
}

std::vector<double> BFGSData::DotInMAndV1(std::vector<std::vector<double>>& matrix, std::vector<double>& vec) 
{
    std::vector<double> result(matrix.size(), 0.0);
    for(int i = 0; i < result.size(); i++)
    {
        for(int j = 0; j < vec.size(); j++)
        {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}
std::vector<double> BFGSData::DotInMAndV2(std::vector<std::vector<double>>& matrix, std::vector<double>& vec) 
{
    std::vector<double> result(matrix.size(), 0.0);
    for(int i = 0; i < result.size(); i++)
    {
        for(int j = 0; j < vec.size(); j++)
        {
            result[i] += matrix[j][i] * vec[j];
        }
    }
    return result;
}

double BFGSData::DotInVAndV(std::vector<double>& vec1, std::vector<double>& vec2) 
{
    double result = 0.0;
    for(int i = 0; i < vec1.size(); i++)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

std::vector<std::vector<double>> BFGSData::OuterVAndV(std::vector<double>& a, std::vector<double>& b) 
{
    std::vector<std::vector<double>> result = std::vector<std::vector<double>>(a.size(), std::vector<double>(b.size(), 0.0));
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < b.size(); j++)
        {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

std::vector<std::vector<double>> BFGSData::ScaleMatrix(std::vector<std::vector<double>>& a, double& b)
{
    std::vector<std::vector<double>> result = std::vector<std::vector<double>>(a.size(), std::vector<double>(a[0].size(), 0.0));
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < a[0].size(); j++)
        {
            result[i][j] = a[i][j] * b;
        }
    }
    return result;
}

std::vector<std::vector<double>> BFGSData::MatAdd(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b)
{
    std::vector<std::vector<double>> result = std::vector<std::vector<double>>(a.size(), std::vector<double>(a[0].size(), 0.0));
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < a[0].size(); j++)
        {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

void BFGSData::RecPrtVec(char* info, std::vector<double>& vec, int nrow, int ncol)
{
    std::cout << "printing data for " << info << std::endl;
    int iloc = 0;
    for(int irow = 0; irow < nrow; ++irow)
    {
        for(int icol = 0; icol < ncol; ++icol)
        {
            std::cout << " " << vec[iloc];
            iloc++; 
        }
        std::cout << std::endl;
    }
}

void BFGSData::RecPrtMat(char* info, std::vector<std::vector<double>>& mat, int nrow, int ncol)
{
    std::cout << "printing data for " << info << std::endl;
    for(int irow = 0; irow < nrow; ++irow)
    {
        for(int icol = 0; icol < ncol; ++icol)
        {
            std::cout << " " << mat[irow][icol];
        }
        std::cout << std::endl;
    }
}

double BFGSData::FindMaxVec(std::vector<double> vec)
{
    std::vector<double>::iterator it = std::max_element(vec.begin(), vec.end());
    double max_value = *it;
    return max_value;
}
