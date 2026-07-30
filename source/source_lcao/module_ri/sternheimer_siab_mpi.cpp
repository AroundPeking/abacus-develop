#include "sternheimer_siab_mpi.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>

namespace module_ri
{
namespace sternheimer_siab
{
namespace
{

using Complex = std::complex<double>;

static_assert(std::numeric_limits<int>::digits <= std::numeric_limits<double>::digits,
              "Sternheimer SIAB MPI packing requires every int value bit to fit in a double mantissa.");

std::vector<ReferenceRow> sorted_rows(std::vector<ReferenceRow> rows)
{
    const auto key = [](const ReferenceRow& row) {
        return std::make_tuple(row.occupied_state, row.auxiliary_channel, row.frequency_index);
    };
    std::sort(rows.begin(), rows.end(), [&key](const ReferenceRow& left, const ReferenceRow& right) {
        return key(left) < key(right);
    });
    for (std::size_t index = 1; index < rows.size(); ++index)
    {
        if (key(rows[index - 1]) == key(rows[index]))
        {
            throw std::invalid_argument("Sternheimer SIAB gathered reference rows contain a duplicate key.");
        }
    }
    return rows;
}

bool valid_local_rows(const std::vector<ReferenceRow>& rows, const std::size_t nprimitive)
{
    for (const ReferenceRow& row: rows)
    {
        if (row.q.size() != nprimitive || row.occupied_state < 0 || row.auxiliary_channel < 0 || row.frequency_index < 0
            || !std::isfinite(row.frequency_ha) || !std::isfinite(row.occupation)
            || !std::isfinite(row.frequency_weight) || !std::isfinite(row.norm))
        {
            return false;
        }
        for (const Complex& value: row.q)
        {
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            {
                return false;
            }
        }
    }
    return true;
}

std::size_t row_width(const std::size_t nprimitive)
{
    if (nprimitive > (std::numeric_limits<std::size_t>::max() - 7) / 2)
    {
        throw std::overflow_error("Sternheimer SIAB reference row width overflows size_t.");
    }
    return 7 + 2 * nprimitive;
}

std::vector<double> pack_rows(const std::vector<ReferenceRow>& rows, const std::size_t nprimitive)
{
    const std::size_t width = row_width(nprimitive);
    std::vector<double> packed;
    packed.reserve(rows.size() * width);
    for (const ReferenceRow& row: rows)
    {
        packed.push_back(static_cast<double>(row.occupied_state));
        packed.push_back(static_cast<double>(row.auxiliary_channel));
        packed.push_back(static_cast<double>(row.frequency_index));
        packed.push_back(row.frequency_ha);
        packed.push_back(row.occupation);
        packed.push_back(row.frequency_weight);
        packed.push_back(row.norm);
        for (const Complex& value: row.q)
        {
            packed.push_back(value.real());
            packed.push_back(value.imag());
        }
    }
    return packed;
}

int unpack_index(const double value)
{
    if (!std::isfinite(value) || value < 0.0 || value > static_cast<double>(std::numeric_limits<int>::max())
        || value != std::floor(value))
    {
        throw std::runtime_error("Sternheimer SIAB gathered reference row contains an invalid integer index.");
    }
    return static_cast<int>(value);
}

std::vector<ReferenceRow> unpack_rows(const std::vector<double>& packed, const std::size_t nprimitive)
{
    const std::size_t width = row_width(nprimitive);
    if (packed.size() % width != 0)
    {
        throw std::runtime_error("Sternheimer SIAB gathered reference row payload has an invalid size.");
    }
    std::vector<ReferenceRow> rows(packed.size() / width);
    for (std::size_t irow = 0; irow != rows.size(); ++irow)
    {
        const std::size_t offset = irow * width;
        ReferenceRow& row = rows[irow];
        row.occupied_state = unpack_index(packed[offset]);
        row.auxiliary_channel = unpack_index(packed[offset + 1]);
        row.frequency_index = unpack_index(packed[offset + 2]);
        row.frequency_ha = packed[offset + 3];
        row.occupation = packed[offset + 4];
        row.frequency_weight = packed[offset + 5];
        row.norm = packed[offset + 6];
        row.q.resize(nprimitive);
        for (std::size_t ie = 0; ie != nprimitive; ++ie)
        {
            row.q[ie] = Complex(packed[offset + 7 + 2 * ie], packed[offset + 8 + 2 * ie]);
        }
    }
    return sorted_rows(std::move(rows));
}

bool source_row_width(const std::size_t nprimitive, std::size_t& width)
{
    if (nprimitive > (std::numeric_limits<std::size_t>::max() - 4) / 2)
    {
        return false;
    }
    width = 4 + 2 * nprimitive;
    return true;
}

bool valid_source_rows(const std::vector<SourceRow>& rows, const std::size_t nprimitive)
{
    for (const SourceRow& row: rows)
    {
        if (row.occupied_state < 0 || row.auxiliary_channel < 0 || !std::isfinite(row.occupation)
            || row.occupation <= 0.0 || !std::isfinite(row.norm) || row.norm <= 0.0
            || row.d.size() != nprimitive)
        {
            return false;
        }
        for (const Complex& value: row.d)
        {
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            {
                return false;
            }
        }
    }
    return true;
}

std::vector<SourceRow> sorted_source_rows(std::vector<SourceRow> rows)
{
    const auto key = [](const SourceRow& row) {
        return std::make_pair(row.occupied_state, row.auxiliary_channel);
    };
    std::sort(rows.begin(), rows.end(), [&key](const SourceRow& left, const SourceRow& right) {
        return key(left) < key(right);
    });
    for (std::size_t index = 1; index < rows.size(); ++index)
    {
        if (key(rows[index - 1]) == key(rows[index]))
        {
            throw std::invalid_argument("Sternheimer SIAB gathered source rows contain a duplicate key.");
        }
    }
    return rows;
}

bool source_payload_size(const std::size_t row_count, const std::size_t width, std::size_t& payload_size)
{
    if (row_count > std::numeric_limits<std::size_t>::max() / width)
    {
        return false;
    }
    payload_size = row_count * width;
    return payload_size <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

std::vector<double> pack_source_rows(const std::vector<SourceRow>& rows, const std::size_t payload_size)
{
    std::vector<double> packed;
    packed.reserve(payload_size);
    for (const SourceRow& row: rows)
    {
        packed.push_back(static_cast<double>(row.occupied_state));
        packed.push_back(static_cast<double>(row.auxiliary_channel));
        packed.push_back(row.occupation);
        packed.push_back(row.norm);
        for (const Complex& value: row.d)
        {
            packed.push_back(value.real());
            packed.push_back(value.imag());
        }
    }
    return packed;
}

int unpack_source_index(const double value)
{
    if (!std::isfinite(value) || value < 0.0 || value > static_cast<double>(std::numeric_limits<int>::max())
        || value != std::floor(value))
    {
        throw std::invalid_argument("Sternheimer SIAB gathered source row contains an invalid integer index.");
    }
    return static_cast<int>(value);
}

std::vector<SourceRow> unpack_source_rows(const std::vector<double>& packed, const std::size_t nprimitive)
{
    std::size_t width = 0;
    if (!source_row_width(nprimitive, width) || packed.size() % width != 0)
    {
        throw std::invalid_argument("Sternheimer SIAB gathered source row payload has an invalid size.");
    }

    std::vector<SourceRow> rows(packed.size() / width);
    for (std::size_t irow = 0; irow != rows.size(); ++irow)
    {
        const std::size_t offset = irow * width;
        SourceRow& row = rows[irow];
        row.occupied_state = unpack_source_index(packed[offset]);
        row.auxiliary_channel = unpack_source_index(packed[offset + 1]);
        row.occupation = packed[offset + 2];
        row.norm = packed[offset + 3];
        row.d.resize(nprimitive);
        for (std::size_t ie = 0; ie != nprimitive; ++ie)
        {
            row.d[ie] = Complex(packed[offset + 4 + 2 * ie], packed[offset + 5 + 2 * ie]);
        }
    }
    if (!valid_source_rows(rows, nprimitive))
    {
        throw std::invalid_argument("Sternheimer SIAB gathered source rows are invalid.");
    }
    return sorted_source_rows(std::move(rows));
}

#ifdef __MPI
enum class SourceGatherStatus : int
{
    ok = 0,
    invalid_payload = 1,
    overflow_capacity = 2,
    length = 3,
    allocation = 4,
    unknown_runtime = 5
};

template <typename Operation>
SourceGatherStatus source_gather_status(const Operation& operation)
{
    try
    {
        operation();
        return SourceGatherStatus::ok;
    }
    catch (const std::bad_alloc&)
    {
        return SourceGatherStatus::allocation;
    }
    catch (const std::length_error&)
    {
        return SourceGatherStatus::length;
    }
    catch (const std::overflow_error&)
    {
        return SourceGatherStatus::overflow_capacity;
    }
    catch (const std::invalid_argument&)
    {
        return SourceGatherStatus::invalid_payload;
    }
    catch (const std::exception&)
    {
        return SourceGatherStatus::unknown_runtime;
    }
    catch (...)
    {
        return SourceGatherStatus::unknown_runtime;
    }
}

SourceGatherStatus source_gather_status_from_code(const int code)
{
    if (code < static_cast<int>(SourceGatherStatus::ok) || code > static_cast<int>(SourceGatherStatus::unknown_runtime))
    {
        return SourceGatherStatus::unknown_runtime;
    }
    return static_cast<SourceGatherStatus>(code);
}

SourceGatherStatus allreduce_source_gather_status(const SourceGatherStatus local_status, MPI_Comm communicator)
{
    int status_code = static_cast<int>(local_status);
    MPI_Allreduce(MPI_IN_PLACE, &status_code, 1, MPI_INT, MPI_MAX, communicator);
    return source_gather_status_from_code(status_code);
}

SourceGatherStatus broadcast_source_gather_status(const SourceGatherStatus root_status,
                                                  const int root,
                                                  MPI_Comm communicator)
{
    int status_code = static_cast<int>(root_status);
    MPI_Bcast(&status_code, 1, MPI_INT, root, communicator);
    return source_gather_status_from_code(status_code);
}

void throw_source_gather_failure(const SourceGatherStatus status, const char* message)
{
    switch (status)
    {
    case SourceGatherStatus::ok:
        return;
    case SourceGatherStatus::invalid_payload:
        throw std::invalid_argument(message);
    case SourceGatherStatus::overflow_capacity:
        throw std::overflow_error(message);
    case SourceGatherStatus::length:
        throw std::length_error(message);
    case SourceGatherStatus::allocation:
        throw std::bad_alloc();
    case SourceGatherStatus::unknown_runtime:
        throw std::runtime_error(message);
    }
    throw std::runtime_error(message);
}
#endif

} // namespace

std::vector<std::vector<Complex>> assemble_full_primitive_grids(const std::vector<PrimitiveSlab>& slabs,
                                                                const int nxy,
                                                                const int nz)
{
    if (nxy <= 0 || nz <= 0 || slabs.empty())
    {
        throw std::invalid_argument("Sternheimer SIAB primitive assembly requires positive grid dimensions and slabs.");
    }
    const std::size_t nprimitive = slabs.front().values.size();
    if (nprimitive == 0)
    {
        throw std::invalid_argument("Sternheimer SIAB primitive assembly requires at least one primitive.");
    }

    std::vector<int> plane_coverage(static_cast<std::size_t>(nz), 0);
    std::vector<std::vector<Complex>> full(
        nprimitive,
        std::vector<Complex>(static_cast<std::size_t>(nxy) * static_cast<std::size_t>(nz), Complex(0.0, 0.0)));
    for (const PrimitiveSlab& slab: slabs)
    {
        if (slab.startz < 0 || slab.nplane < 0 || slab.startz + slab.nplane > nz || slab.values.size() != nprimitive)
        {
            throw std::invalid_argument("Sternheimer SIAB primitive slab metadata is inconsistent.");
        }
        const std::size_t local_size = static_cast<std::size_t>(nxy) * static_cast<std::size_t>(slab.nplane);
        for (const auto& primitive: slab.values)
        {
            if (primitive.size() != local_size)
            {
                throw std::invalid_argument("Sternheimer SIAB primitive slab payload size is inconsistent.");
            }
        }
        for (int iz = 0; iz != slab.nplane; ++iz)
        {
            ++plane_coverage[static_cast<std::size_t>(slab.startz + iz)];
        }
        for (std::size_t ie = 0; ie != nprimitive; ++ie)
        {
            for (int ixy = 0; ixy != nxy; ++ixy)
            {
                for (int iz = 0; iz != slab.nplane; ++iz)
                {
                    full[ie][static_cast<std::size_t>(ixy) * static_cast<std::size_t>(nz)
                             + static_cast<std::size_t>(slab.startz + iz)]
                        = slab.values[ie][static_cast<std::size_t>(ixy) * static_cast<std::size_t>(slab.nplane)
                                          + static_cast<std::size_t>(iz)];
                }
            }
        }
    }
    if (!std::all_of(plane_coverage.begin(), plane_coverage.end(), [](const int count) { return count == 1; }))
    {
        throw std::invalid_argument("Sternheimer SIAB primitive slabs must cover every PW z plane exactly once.");
    }
    return full;
}

#ifdef __MPI
std::vector<std::vector<Complex>> allgather_full_primitive_grids(
    const std::vector<std::vector<Complex>>& local_primitives,
    const int nxy,
    const int nz,
    const int startz,
    const int nplane,
    MPI_Comm communicator)
{
    int rank_count = 0;
    MPI_Comm_size(communicator, &rank_count);
    const bool local_valid
        = nxy > 0 && nz > 0 && startz >= 0 && nplane >= 0 && startz + nplane <= nz && !local_primitives.empty()
          && std::all_of(local_primitives.begin(), local_primitives.end(), [nxy, nplane](const auto& values) {
                 return values.size() == static_cast<std::size_t>(nxy) * static_cast<std::size_t>(nplane);
             });
    int valid = local_valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, communicator);
    if (valid == 0)
    {
        throw std::invalid_argument("Sternheimer SIAB local primitive slab is invalid on at least one MPI rank.");
    }

    const int local_nprimitive = static_cast<int>(local_primitives.size());
    int min_nprimitive = local_nprimitive;
    int max_nprimitive = local_nprimitive;
    MPI_Allreduce(MPI_IN_PLACE, &min_nprimitive, 1, MPI_INT, MPI_MIN, communicator);
    MPI_Allreduce(MPI_IN_PLACE, &max_nprimitive, 1, MPI_INT, MPI_MAX, communicator);
    if (min_nprimitive != max_nprimitive)
    {
        throw std::invalid_argument("Sternheimer SIAB primitive counts differ between MPI ranks.");
    }

    const std::size_t local_grid_size = static_cast<std::size_t>(nxy) * static_cast<std::size_t>(nplane);
    const std::size_t local_payload_size = local_grid_size * local_primitives.size();
    if (local_payload_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("Sternheimer SIAB local primitive payload exceeds MPI int count capacity.");
    }
    std::vector<Complex> local_payload;
    local_payload.reserve(local_payload_size);
    for (const auto& primitive: local_primitives)
    {
        local_payload.insert(local_payload.end(), primitive.begin(), primitive.end());
    }

    std::vector<int> startz_by_rank(static_cast<std::size_t>(rank_count));
    std::vector<int> nplane_by_rank(static_cast<std::size_t>(rank_count));
    std::vector<int> counts(static_cast<std::size_t>(rank_count));
    const int local_count = static_cast<int>(local_payload_size);
    MPI_Allgather(&startz, 1, MPI_INT, startz_by_rank.data(), 1, MPI_INT, communicator);
    MPI_Allgather(&nplane, 1, MPI_INT, nplane_by_rank.data(), 1, MPI_INT, communicator);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator);

    std::vector<int> displacements(static_cast<std::size_t>(rank_count), 0);
    std::size_t total_count = 0;
    for (int rank = 0; rank != rank_count; ++rank)
    {
        if (total_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("Sternheimer SIAB primitive allgather exceeds MPI int displacement capacity.");
        }
        displacements[static_cast<std::size_t>(rank)] = static_cast<int>(total_count);
        total_count += static_cast<std::size_t>(counts[static_cast<std::size_t>(rank)]);
    }
    std::vector<Complex> gathered(total_count);
    MPI_Allgatherv(local_payload.data(),
                   local_count,
                   MPI_DOUBLE_COMPLEX,
                   gathered.data(),
                   counts.data(),
                   displacements.data(),
                   MPI_DOUBLE_COMPLEX,
                   communicator);

    std::vector<PrimitiveSlab> slabs(static_cast<std::size_t>(rank_count));
    for (int rank = 0; rank != rank_count; ++rank)
    {
        PrimitiveSlab& slab = slabs[static_cast<std::size_t>(rank)];
        slab.startz = startz_by_rank[static_cast<std::size_t>(rank)];
        slab.nplane = nplane_by_rank[static_cast<std::size_t>(rank)];
        const std::size_t rank_grid_size = static_cast<std::size_t>(nxy) * static_cast<std::size_t>(slab.nplane);
        slab.values.resize(static_cast<std::size_t>(local_nprimitive));
        for (int ie = 0; ie != local_nprimitive; ++ie)
        {
            const std::size_t begin = static_cast<std::size_t>(displacements[static_cast<std::size_t>(rank)])
                                      + static_cast<std::size_t>(ie) * rank_grid_size;
            slab.values[static_cast<std::size_t>(ie)].assign(gathered.begin() + begin,
                                                             gathered.begin() + begin + rank_grid_size);
        }
    }
    return assemble_full_primitive_grids(slabs, nxy, nz);
}

std::vector<ReferenceRow> gather_reference_rows_to_root(const std::vector<ReferenceRow>& local_rows,
                                                        const std::size_t nprimitive,
                                                        const int root,
                                                        MPI_Comm communicator)
{
    int rank = 0;
    int rank_count = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &rank_count);
    if (root < 0 || root >= rank_count)
    {
        throw std::invalid_argument("Sternheimer SIAB row gather root is out of range.");
    }
    int valid = valid_local_rows(local_rows, nprimitive) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, communicator);
    if (valid == 0)
    {
        throw std::invalid_argument("Sternheimer SIAB local reference rows are invalid on at least one MPI rank.");
    }

    const std::vector<double> local_packed = pack_rows(local_rows, nprimitive);
    if (local_packed.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("Sternheimer SIAB local row payload exceeds MPI int count capacity.");
    }
    const int local_count = static_cast<int>(local_packed.size());
    std::vector<int> counts(rank == root ? static_cast<std::size_t>(rank_count) : 0);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, root, communicator);

    std::vector<int> displacements;
    std::vector<double> gathered;
    if (rank == root)
    {
        displacements.assign(static_cast<std::size_t>(rank_count), 0);
        std::size_t total_count = 0;
        for (int source_rank = 0; source_rank != rank_count; ++source_rank)
        {
            if (total_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::overflow_error("Sternheimer SIAB row gather exceeds MPI int displacement capacity.");
            }
            displacements[static_cast<std::size_t>(source_rank)] = static_cast<int>(total_count);
            total_count += static_cast<std::size_t>(counts[static_cast<std::size_t>(source_rank)]);
        }
        gathered.resize(total_count);
    }
    MPI_Gatherv(local_packed.data(),
                local_count,
                MPI_DOUBLE,
                gathered.data(),
                counts.data(),
                displacements.data(),
                MPI_DOUBLE,
                root,
                communicator);
    return rank == root ? unpack_rows(gathered, nprimitive) : std::vector<ReferenceRow>();
}

std::vector<SourceRow> gather_source_rows_to_root(const std::vector<SourceRow>& local_rows,
                                                  const std::size_t nprimitive,
                                                  const int root,
                                                  MPI_Comm communicator)
{
    int rank = 0;
    int rank_count = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &rank_count);

    int root_min = root;
    int root_max = root;
    unsigned long long nprimitive_min = static_cast<unsigned long long>(nprimitive);
    unsigned long long nprimitive_max = nprimitive_min;
    MPI_Allreduce(MPI_IN_PLACE, &root_min, 1, MPI_INT, MPI_MIN, communicator);
    MPI_Allreduce(MPI_IN_PLACE, &root_max, 1, MPI_INT, MPI_MAX, communicator);
    MPI_Allreduce(MPI_IN_PLACE, &nprimitive_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator);
    MPI_Allreduce(MPI_IN_PLACE, &nprimitive_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
    if (root_min != root_max || root_min < 0 || root_min >= rank_count)
    {
        throw std::invalid_argument("Sternheimer SIAB source row gather root is inconsistent or out of range.");
    }
    if (nprimitive_min != nprimitive_max)
    {
        throw std::invalid_argument("Sternheimer SIAB source primitive counts differ between MPI ranks.");
    }

    std::size_t width = 0;
    std::size_t local_payload_size = 0;
    const bool width_valid = source_row_width(nprimitive, width);
    SourceGatherStatus local_status = SourceGatherStatus::ok;
    if (!width_valid || !source_payload_size(local_rows.size(), width, local_payload_size))
    {
        local_status = SourceGatherStatus::overflow_capacity;
    }
    else if (!valid_source_rows(local_rows, nprimitive))
    {
        local_status = SourceGatherStatus::invalid_payload;
    }
    SourceGatherStatus collective_status = allreduce_source_gather_status(local_status, communicator);
    throw_source_gather_failure(collective_status,
                                "Sternheimer SIAB source row validation failed on at least one MPI rank.");

    std::vector<double> local_packed;
    local_status = source_gather_status([&local_packed, &local_rows, local_payload_size]() {
        local_packed = pack_source_rows(local_rows, local_payload_size);
    });
    collective_status = allreduce_source_gather_status(local_status, communicator);
    throw_source_gather_failure(collective_status,
                                "Sternheimer SIAB source row packing failed on at least one MPI rank.");
    const int local_count = static_cast<int>(local_packed.size());

    std::vector<int> counts;
    SourceGatherStatus root_status = SourceGatherStatus::ok;
    if (rank == root)
    {
        root_status
            = source_gather_status([&counts, rank_count]() { counts.resize(static_cast<std::size_t>(rank_count)); });
    }
    collective_status = broadcast_source_gather_status(root_status, root, communicator);
    throw_source_gather_failure(collective_status,
                                "Sternheimer SIAB source row count allocation failed on the root MPI rank.");

    MPI_Gather(&local_count, 1, MPI_INT, rank == root ? counts.data() : nullptr, 1, MPI_INT, root, communicator);

    std::vector<int> displacements;
    std::vector<double> gathered;
    root_status = SourceGatherStatus::ok;
    if (rank == root)
    {
        root_status = source_gather_status([&counts, &displacements, &gathered, rank_count, width]() {
            displacements.assign(static_cast<std::size_t>(rank_count), 0);
            std::size_t total_count = 0;
            for (int source_rank = 0; source_rank != rank_count; ++source_rank)
            {
                const int count = counts[static_cast<std::size_t>(source_rank)];
                if (count < 0 || static_cast<std::size_t>(count) % width != 0)
                {
                    throw std::invalid_argument(
                        "Sternheimer SIAB source row gather received an invalid rank payload size.");
                }
                if (static_cast<std::size_t>(count)
                    > static_cast<std::size_t>(std::numeric_limits<int>::max()) - total_count)
                {
                    throw std::overflow_error("Sternheimer SIAB source row gather exceeds MPI displacement capacity.");
                }
                displacements[static_cast<std::size_t>(source_rank)] = static_cast<int>(total_count);
                total_count += static_cast<std::size_t>(count);
            }
            gathered.resize(total_count);
        });
    }
    collective_status = broadcast_source_gather_status(root_status, root, communicator);
    throw_source_gather_failure(collective_status,
                                "Sternheimer SIAB source row gather layout failed on the root MPI rank.");

    MPI_Gatherv(local_packed.empty() ? nullptr : local_packed.data(),
                local_count,
                MPI_DOUBLE,
                rank == root && !gathered.empty() ? gathered.data() : nullptr,
                rank == root ? counts.data() : nullptr,
                rank == root ? displacements.data() : nullptr,
                MPI_DOUBLE,
                root,
                communicator);

    std::vector<SourceRow> result;
    root_status = SourceGatherStatus::ok;
    if (rank == root)
    {
        root_status = source_gather_status(
            [&result, &gathered, nprimitive]() { result = unpack_source_rows(gathered, nprimitive); });
    }
    collective_status = broadcast_source_gather_status(root_status, root, communicator);
    throw_source_gather_failure(collective_status,
                                "Sternheimer SIAB gathered source row unpack failed on the root MPI rank.");
    return rank == root ? result : std::vector<SourceRow>();
}
#else
std::vector<std::vector<Complex>> allgather_full_primitive_grids(
    const std::vector<std::vector<Complex>>& local_primitives,
    const int nxy,
    const int nz,
    const int startz,
    const int nplane)
{
    PrimitiveSlab slab;
    slab.startz = startz;
    slab.nplane = nplane;
    slab.values = local_primitives;
    return assemble_full_primitive_grids({slab}, nxy, nz);
}

std::vector<ReferenceRow> gather_reference_rows_to_root(const std::vector<ReferenceRow>& local_rows,
                                                        const std::size_t nprimitive,
                                                        const int root)
{
    if (root != 0 || !valid_local_rows(local_rows, nprimitive))
    {
        throw std::invalid_argument("Sternheimer SIAB serial reference row gather input is invalid.");
    }
    return sorted_rows(local_rows);
}

std::vector<SourceRow> gather_source_rows_to_root(const std::vector<SourceRow>& local_rows,
                                                  const std::size_t nprimitive,
                                                  const int root)
{
    std::size_t width = 0;
    if (root != 0)
    {
        throw std::invalid_argument("Sternheimer SIAB serial source row gather requires root zero.");
    }
    if (!source_row_width(nprimitive, width))
    {
        throw std::overflow_error("Sternheimer SIAB source row width overflows size_t.");
    }
    if (!valid_source_rows(local_rows, nprimitive))
    {
        throw std::invalid_argument("Sternheimer SIAB serial source rows are invalid.");
    }
    return sorted_source_rows(local_rows);
}
#endif

} // namespace sternheimer_siab
} // namespace module_ri
