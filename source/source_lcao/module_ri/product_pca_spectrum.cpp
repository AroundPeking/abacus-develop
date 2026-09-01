#include "product_pca_spectrum.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace ABFs_Construct
{
namespace PCA
{
namespace
{

bool env_flag_enabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool is_near_threshold(const double eigenvalue, const double threshold)
{
    return threshold > 0.0 && eigenvalue >= 0.5 * threshold && eigenvalue <= 2.0 * threshold;
}

} // namespace

std::vector<ProductPcaTypeSummary> summarize_product_pca_spectrum(const ProductPcaEigenvalues& eigenvalues,
                                                                  const double threshold_ratio)
{
    if (threshold_ratio < 0.0)
    {
        throw std::invalid_argument("product-PCA threshold ratio must be non-negative");
    }

    std::vector<ProductPcaTypeSummary> result(eigenvalues.size());
    for (std::size_t T = 0; T != eigenvalues.size(); ++T)
    {
        ProductPcaTypeSummary& type = result[T];
        type.type = T;
        for (const auto& channel: eigenvalues[T])
        {
            for (const double eigenvalue: channel)
            {
                type.global_max = std::max(type.global_max, eigenvalue);
            }
        }
        type.threshold = type.global_max * threshold_ratio;
        type.channels.resize(eigenvalues[T].size());

        for (std::size_t L = 0; L != eigenvalues[T].size(); ++L)
        {
            const std::vector<double>& channel = eigenvalues[T][L];
            ProductPcaChannelSummary& summary = type.channels[L];
            summary.angular_momentum = L;
            summary.total_count = channel.size();
            if (channel.empty())
            {
                continue;
            }

            summary.min_eigenvalue = *std::min_element(channel.begin(), channel.end());
            summary.max_eigenvalue = *std::max_element(channel.begin(), channel.end());
            double min_positive = std::numeric_limits<double>::infinity();
            for (const double eigenvalue: channel)
            {
                if (eigenvalue > type.threshold)
                {
                    ++summary.retained_count;
                }
                if (is_near_threshold(eigenvalue, type.threshold))
                {
                    ++summary.near_threshold_count;
                }
                if (eigenvalue > 0.0)
                {
                    min_positive = std::min(min_positive, eigenvalue);
                }
            }
            summary.condition_number = std::isfinite(min_positive) ? summary.max_eigenvalue / min_positive
                                                                   : std::numeric_limits<double>::infinity();
        }
    }
    return result;
}

void write_product_pca_spectrum(std::ostream& output,
                                const ProductPcaEigenvalues& eigenvalues,
                                const double threshold_ratio)
{
    const auto summary = summarize_product_pca_spectrum(eigenvalues, threshold_ratio);
    output << std::setprecision(17);
    output << "format\tproduct_pca_spectrum_v1\n";
    output << "threshold_ratio\t" << threshold_ratio << '\n';
    output << "near_threshold_window\t0.5\t2\n";
    for (const ProductPcaTypeSummary& type: summary)
    {
        output << "type\t" << type.type << "\tglobal_max\t" << type.global_max << "\tthreshold\t" << type.threshold
               << '\n';
        for (const ProductPcaChannelSummary& channel: type.channels)
        {
            output << "channel\t" << type.type << '\t' << channel.angular_momentum << "\ttotal\t" << channel.total_count
                   << "\tretained\t" << channel.retained_count << "\tnear_threshold\t" << channel.near_threshold_count
                   << "\tmin\t" << channel.min_eigenvalue << "\tmax\t" << channel.max_eigenvalue << "\tcondition\t"
                   << channel.condition_number << '\n';
            const auto& values = eigenvalues[type.type][channel.angular_momentum];
            for (std::size_t index = 0; index != values.size(); ++index)
            {
                const double value = values[index];
                output << "eigenvalue\t" << type.type << '\t' << channel.angular_momentum << '\t' << index << '\t'
                       << value << "\tretained\t" << (value > type.threshold ? 1 : 0) << "\tnear_threshold\t"
                       << (is_near_threshold(value, type.threshold) ? 1 : 0) << '\n';
            }
        }
    }
}

void write_product_pca_spectrum_file(const std::string& path,
                                     const ProductPcaEigenvalues& eigenvalues,
                                     const double threshold_ratio)
{
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("cannot open product-PCA spectrum diagnostic: " + path);
    }
    write_product_pca_spectrum(output, eigenvalues, threshold_ratio);
}

bool product_pca_spectrum_diagnostic_only_enabled()
{
    return env_flag_enabled("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_ONLY");
}

bool product_pca_spectrum_diagnostic_enabled()
{
    return env_flag_enabled("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG") || product_pca_spectrum_diagnostic_only_enabled();
}

bool product_pca_spectrum_should_stop_after_generation(const std::string& calculation)
{
    return product_pca_spectrum_diagnostic_only_enabled() && calculation == "gen_opt_abfs";
}

std::string product_pca_spectrum_output_path()
{
    const char* path = std::getenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_FILE");
    return path != nullptr && path[0] != '\0' ? path : "PRODUCT_PCA_SPECTRUM.dat";
}

} // namespace PCA
} // namespace ABFs_Construct
