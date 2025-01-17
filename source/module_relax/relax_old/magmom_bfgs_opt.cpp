#include "magmom_bfgs_opt.h"
#include "bfgsdata.h"
#include <string>
#include "module_hamilt_lcao/module_deltaspin/spin_constrain.h"
//#include "module_hamilt_lcao/module_deltaspin/spin_constrain.cpp"

Magmom_BFGS_Opt::Magmom_BFGS_Opt(){}
Magmom_BFGS_Opt::~Magmom_BFGS_Opt(){}

//! initialize H0、H、pos0、grad0、grad
void Magmom_BFGS_Opt::initialize(const int _n_atom, const int _n_moment_component) 
{
    n_atom = _n_atom;
    n_moment_component = _n_moment_component;
    int size = n_atom * n_moment_component;


//    mag_moment = std::vector<std::vector<double>> (n_atom, std::vector<double>(n_moment_component, 0.0));
//    mag_force = std::vector<std::vector<double>> (n_atom, std::vector<double>(n_moment_component, 0.0));
    delta_moment = std::vector<std::vector<double>> (n_atom, std::vector<double>(n_moment_component, 0.0));
    new_moment = std::vector<std::vector<double>> (n_atom, std::vector<double>(n_moment_component, 0.0));

    vec_delta_moment = std::vector<double> (size, 0.0);
    vec_mag_moment = std::vector<double> (size, 0.0);
    vec_mag_force = std::vector<double> (size, 0.0);
    module_moment = std::vector<double> (n_atom, 0.0);
    module_force = std::vector<double> (n_atom, 0.0);

    max_delta_moment = 0.2; // setting 0.2 as our initial trial. You definitely do not want the magnetic moment to change by 1.0 in a step.
    converged = false;
    convergence_threshold = 1.0e-5; // setting 1E-5 as an initial trial. Will change this number later. 
    BFGSData::allocate(size);
}

bool Magmom_BFGS_Opt::bfgs_wrapper()
{
    std::cout << "hello from bfgs_wrapper" << std::endl;
    std::cout << "obtainging the sc instance" << std::endl;
    spinconstrain::SpinConstrain<std::complex<double>>& sc = spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();

    std::cout << "obtaining lambda" << std::endl;
    auto& vec3_magforce = sc.get_sc_lambda();
    std::cout << "obtaining target magnetic moment" << std::endl;
    auto& vec3_magmoment = sc.get_target_mag();

    std::cout << "converting from matrix_vector3 to matrix, n_atom = " << n_atom << std::endl;
    std::vector<std::vector<double>> magforce = matvec3_to_mat(vec3_magforce, n_atom);
    std::cout << "conversion of magforce done" << std::endl;
    std::vector<std::vector<double>> magmoment = matvec3_to_mat(vec3_magmoment, n_atom);

    std::cout << "obtaining magmoment constraints" << std::endl;
    std::vector<ModuleBase::Vector3<int>> vec3_magconstrain = sc.get_constrain();

    double factor = -1.0;
    magforce = BFGSData::ScaleMatrix(magforce, factor);

    std::string str = "read-in magforce";
    BFGSData::RecPrtMat(str, magforce, n_atom, n_moment_component);

//    constrain_magforce(vec3_magforce, vec3_magconstrain);
    str = "read-in magmoment";
    BFGSData::RecPrtMat(str, magmoment, n_atom, n_moment_component);
    magmoment = matvec3_to_mat(vec3_magmoment, n_atom);
    str = "magmoment after bfgs";
    BFGSData::RecPrtMat(str, magmoment, n_atom, n_moment_component);

    std::vector<std::vector<double>> new_magmom = calc_new_magmom(magmoment, magforce);
    str = "new magmoment";
    BFGSData::RecPrtMat(str, magmoment, n_atom, n_moment_component);

    std::cout << "converting matrix back to matrix_vec3" << std::endl;
    const std::vector<ModuleBase::Vector3<double>> vec3_newmag = mat_to_matvec3(new_magmom, n_atom);
    std::cout << "converting matrix back to matrix_vec3 done" << std::endl;
    sc.set_target_mag(vec3_newmag);
    sc.set_read_target_mag(false);
    std::cout << "setting new magnetic moments" << std::endl;

    std::cout << "convergence status = " << get_convergence_status() << std::endl;
    return get_convergence_status(); 
}

void Magmom_BFGS_Opt::constrain_magforce(std::vector<ModuleBase::Vector3<double>>& magforce, std::vector<ModuleBase::Vector3<int>> magconstrain)
{
    int ncol = 3;
    for (size_t i = 0; i < magforce.size(); ++i)
    {
        for (size_t j = 0; j < ncol; ++j)
        {
            if (magconstrain[i][j] == 0) { magforce[i][j] = 0.0; } 
        }
    }
}

std::vector<std::vector<double>> Magmom_BFGS_Opt::calc_new_magmom(std::vector<std::vector<double>> _magmom, std::vector<std::vector<double>> _magforce)
{
//    std::cout << "checking convergence" << std::endl;
    module_force = calc_moment_module(_magforce, n_atom, n_moment_component);
    double max_force_component = FindAbsMaxVec(module_force);
//    std::cout << "maximum of force component: " << max_force_component << std::endl;
    if (max_force_component < convergence_threshold)
    {
        converged = true;
    } else
    {
        converged = false;
    }
//  because the current code performs synchronized optimization of magnetic moment and geometry
//  optimziation, the optimization iteration may still continue even when the magnetic moment
//  is converged. 
//the BFGS procedure is performed even when the convergence is acheived for magnetic moment.

//  First, obtain the change in the magnetic moment using the relax_step function in BFGSData
//    std::cout << "converting matrix to vectors for BFGS calculations" << std::endl; 
    vec_mag_moment = mat_to_vec(_magmom, n_atom, n_moment_component); 
    vec_mag_force = mat_to_vec(_magforce, n_atom, n_moment_component);

    BFGSData::RecPrtVec("Converted Moment Vector", vec_mag_moment, n_atom, n_moment_component);
    BFGSData::RecPrtVec("Converted Force  Vector", vec_mag_force,  n_atom, n_moment_component);
    vec_delta_moment = BFGSData::relax_step(vec_mag_moment, vec_mag_force);
//    BFGSData::RecPrtVec("Change in Moment", vec_delta_moment, n_atom, n_moment_component);


//  Then check if the step is greater than the threshold;
    delta_moment = vec_to_mat(vec_delta_moment, n_atom, n_moment_component);
    module_moment = calc_moment_module(delta_moment, n_atom, n_moment_component);
    double max_moment_component = FindAbsMaxVec(module_moment);

    double scaling_factor;
    if (max_moment_component > max_delta_moment)
    {
        scaling_factor = -1.0*max_delta_moment/max_moment_component;
    } else
    {
        scaling_factor = -1.0;
    }
//  At last, subtract new moment from the input moment
    delta_moment = BFGSData::ScaleMatrix(delta_moment, scaling_factor);
    new_moment = BFGSData::MatAdd(_magmom, delta_moment);
    return new_moment;
}


std::vector<double> Magmom_BFGS_Opt::calc_moment_module(std::vector<std::vector<double>> mat_moment, int nrow, int ncol)
{
    std::vector<double> result_vec(nrow, 0.0);
    double temp_double;
    for (int irow = 0; irow < nrow; ++irow)
    {
        temp_double = 0.0;
        for (int icol = 0; icol < ncol; ++icol)
        {
            temp_double += mat_moment[irow][icol] * mat_moment[irow][icol];
        }
        result_vec[irow] = std::sqrt(temp_double); 
    }
    return result_vec;
}


std::vector<double> Magmom_BFGS_Opt::mat_to_vec(std::vector<std::vector<double>> mat, int nrow, int ncol)
{
    int iloc = 0;
    int nelem = nrow * ncol;
    std::vector<double> vec(nelem);
    for (int irow = 0; irow < nrow; ++irow)
    {
        for (int icol = 0; icol < ncol; ++icol)
        {
            vec[iloc] = mat[irow][icol];
            iloc++;
        }
    }
    return vec;
}

std::vector<std::vector<double>> Magmom_BFGS_Opt::vec_to_mat(std::vector<double> vec, int nrow, int ncol)
{
    int iloc = 0;
    int nelem = nrow * ncol;
    std::vector<std::vector<double>> mat(nrow, std::vector<double>(ncol));
    for (int irow = 0; irow < nrow; ++irow)
    {
        for (int icol = 0; icol < ncol; ++icol)
        {
            mat[irow][icol] = vec[iloc];
            iloc++;
        }
    }
    return mat;
}

void Magmom_BFGS_Opt::set_convergence_threshold(double _convergence_threshold)
{
    convergence_threshold = _convergence_threshold;
}

void Magmom_BFGS_Opt::set_max_delta_moment(double _max_delta_moment)
{
    max_delta_moment = _max_delta_moment;
}

bool Magmom_BFGS_Opt::get_convergence_status()
{
    return converged;
}

double Magmom_BFGS_Opt::FindAbsMaxVec(std::vector<double> vec)
{
    if (vec.empty()) {
        return 0.0;
    }
    double max_value = std::abs(vec[0]);
    double abs_val;
    for (size_t i = 1; i < vec.size(); ++i)
    {
        abs_val = std::abs(vec[i]);
        if (max_value < abs_val) { max_value = abs_val; }
    }
    return max_value;
}

std::vector<std::vector<double>> Magmom_BFGS_Opt::matvec3_to_mat(std::vector<ModuleBase::Vector3<double>> matvec3, int nrow)
{
    int ncol = 3;
    std::vector<std::vector<double>> mat(nrow, std::vector<double>(ncol));

    for (int irow = 0; irow < nrow; ++irow){
        for (int icol = 0; icol < ncol; ++icol){
            mat[irow][icol] = matvec3[irow][icol];
        }
    }
    return mat;
}


std::vector<ModuleBase::Vector3<double>> Magmom_BFGS_Opt::mat_to_matvec3(std::vector<std::vector<double>> mat, int nrow)
{
    int ncol = 3;
    std::cout << "entering mat_to_matvec3" << std::endl;
    std::vector<ModuleBase::Vector3<double>> matvec3(nrow);
    std::cout << "temporary matvec3 created" << std::endl;

    std::cout << mat[0][0] <<  std::endl;
    for (int irow = 0; irow < nrow; ++irow){
        for (int icol = 0; icol < ncol; ++icol){
            std::cout << irow << " " << icol << std::endl;
            matvec3[irow][icol] = mat[irow][icol];
            std::cout << matvec3[irow][icol] << " " << mat[irow][icol] << std::endl; 
        }
    }
    return matvec3;
}

