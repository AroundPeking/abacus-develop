#ifndef PRODUCT_PCA_SPECTRUM_H
#define PRODUCT_PCA_SPECTRUM_H

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace ABFs_Construct
{
namespace PCA
{

using ProductPcaEigenvalues = std::vector<std::vector<std::vector<double>>>;

struct ProductPcaChannelSummary
{
    std::size_t angular_momentum = 0;
    std::size_t total_count = 0;
    std::size_t retained_count = 0;
    std::size_t near_threshold_count = 0;
    double min_eigenvalue = 0.0;
    double max_eigenvalue = 0.0;
    double condition_number = 0.0;
};

struct ProductPcaTypeSummary
{
    std::size_t type = 0;
    double global_max = 0.0;
    double threshold = 0.0;
    std::vector<ProductPcaChannelSummary> channels;
};

std::vector<ProductPcaTypeSummary> summarize_product_pca_spectrum(const ProductPcaEigenvalues& eigenvalues,
                                                                  double threshold_ratio);

void write_product_pca_spectrum(std::ostream& output, const ProductPcaEigenvalues& eigenvalues, double threshold_ratio);

void write_product_pca_spectrum_file(const std::string& path,
                                     const ProductPcaEigenvalues& eigenvalues,
                                     double threshold_ratio);

bool product_pca_spectrum_diagnostic_enabled();
bool product_pca_spectrum_diagnostic_only_enabled();
bool product_pca_spectrum_should_stop_after_generation(const std::string& calculation);
std::string product_pca_spectrum_output_path();

} // namespace PCA
} // namespace ABFs_Construct

#endif // PRODUCT_PCA_SPECTRUM_H
