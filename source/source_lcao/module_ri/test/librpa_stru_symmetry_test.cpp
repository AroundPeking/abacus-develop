#include "source_lcao/module_ri/librpa_stru_symmetry.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

std::vector<std::string> split_lines(const std::string& text)
{
    std::vector<std::string> lines;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line))
    {
        lines.push_back(line);
    }
    return lines;
}

void require_spatial_row(const std::string& line)
{
    std::istringstream input(line);
    std::vector<std::string> fields;
    std::string field;
    while (input >> field)
    {
        fields.push_back(field);
    }
    require(fields.size() == 12, "a symmetry row must contain 9 integer and 3 translation fields");
    for (int i = 0; i < 9; ++i)
    {
        std::size_t consumed = 0;
        (void)std::stoi(fields[static_cast<std::size_t>(i)], &consumed);
        require(consumed == fields[static_cast<std::size_t>(i)].size(), "rotation fields must be integers");
    }
    for (int i = 9; i < 12; ++i)
    {
        std::size_t consumed = 0;
        const double value = std::stod(fields[static_cast<std::size_t>(i)], &consumed);
        require(consumed == fields[static_cast<std::size_t>(i)].size() && std::isfinite(value),
                "translation fields must be finite floating-point values");
    }
}

void test_magnetic_block()
{
    ModuleSymmetry::Symmetry symmetry;
    symmetry.nrotk = 2;
    symmetry.nrotk_anti = 1;
    symmetry.magnetic_nspin4 = true;
    symmetry.gmatrix[0] = ModuleBase::Matrix3();
    symmetry.gmatrix[1] = ModuleBase::Matrix3(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    symmetry.gtrans[1] = ModuleBase::Vector3<double>(0.5, 0.0, 0.0);
    symmetry.gmatrix_anti[0] = ModuleBase::Matrix3(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    symmetry.gtrans_anti[0] = ModuleBase::Vector3<double>(0.125, 0.25, 0.375);

    std::ostringstream output;
    RpaLriDetail::write_librpa_symmetry_block(output, symmetry, 4);
    const auto lines = split_lines(output.str());
    require(lines.size() == 8, "magnetic SOC block must contain 3 spatial and 3 spin rows");
    require(lines[0] == "3 row", "unitary and antiunitary operations share one spatial block");
    require_spatial_row(lines[1]);
    require_spatial_row(lines[2]);
    require_spatial_row(lines[3]);
    require(lines[3].find("1.250000000000000e-01") != std::string::npos,
            "antiunitary spatial operation must follow unitary operations");
    require(lines[4] == "spin_symmetry 0 2", "LibRPA must reconstruct the magnetic spin action");
    require(lines[5] == "0" && lines[6] == "0" && lines[7] == "1", "only the final operation is antiunitary");
}

void test_nonmagnetic_soc_block()
{
    ModuleSymmetry::Symmetry symmetry;
    symmetry.nrotk = 1;
    symmetry.magnetic_nspin4 = false;
    symmetry.gmatrix[0] = ModuleBase::Matrix3();

    std::ostringstream output;
    RpaLriDetail::write_librpa_symmetry_block(output, symmetry, 4);
    const auto lines = split_lines(output.str());
    require(lines.size() == 4, "nonmagnetic SOC block must contain one spatial and one spin row");
    require(lines[0] == "1 row", "nonmagnetic block must retain its spatial operation");
    require(lines[2] == "spin_symmetry 1 2", "grey-group SOC metadata must be preserved");
    require(lines[3] == "0", "the nonmagnetic spatial operation must be unitary");
}
} // namespace

int main()
{
    test_magnetic_block();
    test_nonmagnetic_soc_block();
    std::cout << "LibRPA stru_out symmetry tests passed\n";
    return 0;
}
