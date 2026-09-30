#ifndef LIBRPA_STRU_SYMMETRY_H
#define LIBRPA_STRU_SYMMETRY_H

#include "source_base/matrix3.h"
#include "source_base/vector3.h"
#include "source_cell/module_symmetry/symmetry.h"

#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace RpaLriDetail
{
inline int checked_near_int(const double value, const std::string& context)
{
    const double rounded = std::round(value);
    if (std::abs(value - rounded) > 1e-8)
    {
        throw std::runtime_error(context + " is not close to an integer.");
    }
    return static_cast<int>(rounded);
}

// LibRPA derives the SOC spin rotation from each spatial operation. ABACUS
// therefore writes one spatial table for both ordinary and magnetic groups,
// followed by the antiunitary flags required to reconstruct that action.
inline void write_librpa_symmetry_block(std::ostream& output, const ModuleSymmetry::Symmetry& symmetry, const int nspin)
{
    if (symmetry.nrotk <= 0)
    {
        return;
    }

    const int n_anti = symmetry.magnetic_nspin4 ? symmetry.nrotk_anti : 0;
    const auto write_operation
        = [&output](const ModuleBase::Matrix3& rotation, const ModuleBase::Vector3<double>& translation) {
              output << std::setw(4) << checked_near_int(rotation.e11, "symmetry rotation e11") << std::setw(4)
                     << checked_near_int(rotation.e12, "symmetry rotation e12") << std::setw(4)
                     << checked_near_int(rotation.e13, "symmetry rotation e13") << std::setw(4)
                     << checked_near_int(rotation.e21, "symmetry rotation e21") << std::setw(4)
                     << checked_near_int(rotation.e22, "symmetry rotation e22") << std::setw(4)
                     << checked_near_int(rotation.e23, "symmetry rotation e23") << std::setw(4)
                     << checked_near_int(rotation.e31, "symmetry rotation e31") << std::setw(4)
                     << checked_near_int(rotation.e32, "symmetry rotation e32") << std::setw(4)
                     << checked_near_int(rotation.e33, "symmetry rotation e33") << std::setw(24) << std::scientific
                     << std::setprecision(15) << translation.x << std::setw(24) << std::scientific
                     << std::setprecision(15) << translation.y << std::setw(24) << std::scientific
                     << std::setprecision(15) << translation.z << std::endl;
          };

    output << (symmetry.nrotk + n_anti) << " row" << std::endl;
    for (int isym = 0; isym < symmetry.nrotk; ++isym)
    {
        write_operation(symmetry.gmatrix[isym], symmetry.gtrans[isym]);
    }
    for (int isym = 0; isym < n_anti; ++isym)
    {
        write_operation(symmetry.gmatrix_anti[isym], symmetry.gtrans_anti[isym]);
    }

    if (nspin == 4)
    {
        // source=2 means LibRPA reconstructs U_s from det(Q) Q.
        output << "spin_symmetry " << (symmetry.magnetic_nspin4 ? 0 : 1) << " 2" << std::endl;
        for (int isym = 0; isym < symmetry.nrotk; ++isym)
        {
            output << 0 << std::endl;
        }
        for (int isym = 0; isym < n_anti; ++isym)
        {
            output << 1 << std::endl;
        }
    }
}
} // namespace RpaLriDetail

#endif
