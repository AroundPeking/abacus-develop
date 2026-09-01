#include "../product_pca_spectrum.h"

#include <cstdlib>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

namespace
{

using ABFs_Construct::PCA::ProductPcaEigenvalues;
using ABFs_Construct::PCA::summarize_product_pca_spectrum;

TEST(ProductPcaSpectrum, AppliesOneGlobalThresholdPerType)
{
    const ProductPcaEigenvalues eigenvalues = {{{1.0, 4.0}, {0.0002, 0.004, 2.0}}, {{0.01}, {0.5, 1.0}}};

    const auto summary = summarize_product_pca_spectrum(eigenvalues, 1.0e-3);

    ASSERT_EQ(summary.size(), 2);
    EXPECT_DOUBLE_EQ(summary[0].global_max, 4.0);
    EXPECT_DOUBLE_EQ(summary[0].threshold, 0.004);
    ASSERT_EQ(summary[0].channels.size(), 2);
    EXPECT_EQ(summary[0].channels[0].retained_count, 2);
    EXPECT_EQ(summary[0].channels[1].retained_count, 1);
    EXPECT_EQ(summary[0].channels[1].near_threshold_count, 1);
    EXPECT_DOUBLE_EQ(summary[0].channels[1].min_eigenvalue, 0.0002);
    EXPECT_DOUBLE_EQ(summary[0].channels[1].max_eigenvalue, 2.0);
    EXPECT_DOUBLE_EQ(summary[0].channels[1].condition_number, 10000.0);

    EXPECT_DOUBLE_EQ(summary[1].global_max, 1.0);
    EXPECT_DOUBLE_EQ(summary[1].threshold, 0.001);
    EXPECT_EQ(summary[1].channels[0].retained_count, 1);
    EXPECT_EQ(summary[1].channels[1].retained_count, 2);
}

TEST(ProductPcaSpectrum, WritesEveryPrethresholdEigenvalueDeterministically)
{
    const ProductPcaEigenvalues eigenvalues = {{{0.25, 2.0}, {0.001, 0.01}}};
    std::ostringstream output;

    ABFs_Construct::PCA::write_product_pca_spectrum(output, eigenvalues, 1.0e-2);

    const std::string text = output.str();
    EXPECT_NE(text.find("format\tproduct_pca_spectrum_v1\n"), std::string::npos);
    EXPECT_NE(text.find("type\t0\tglobal_max\t2"), std::string::npos);
    EXPECT_NE(text.find("channel\t0\t1\ttotal\t2\tretained\t0\tnear_threshold\t1"), std::string::npos);
    EXPECT_NE(text.find("eigenvalue\t0\t0\t0\t0.25\tretained\t1\tnear_threshold\t0"), std::string::npos);
    EXPECT_NE(text.find("eigenvalue\t0\t1\t1\t0.01\tretained\t0\tnear_threshold\t1"), std::string::npos);
}

TEST(ProductPcaSpectrum, DiagnosticIsDisabledByDefaultAndAcceptsExplicitTrueOnly)
{
    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG");
    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_ONLY");
    EXPECT_FALSE(ABFs_Construct::PCA::product_pca_spectrum_diagnostic_enabled());
    EXPECT_FALSE(ABFs_Construct::PCA::product_pca_spectrum_diagnostic_only_enabled());

    setenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG", "false", 1);
    EXPECT_FALSE(ABFs_Construct::PCA::product_pca_spectrum_diagnostic_enabled());
    setenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG", "1", 1);
    EXPECT_TRUE(ABFs_Construct::PCA::product_pca_spectrum_diagnostic_enabled());

    setenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_ONLY", "yes", 1);
    EXPECT_TRUE(ABFs_Construct::PCA::product_pca_spectrum_diagnostic_only_enabled());
    EXPECT_TRUE(ABFs_Construct::PCA::product_pca_spectrum_diagnostic_enabled());

    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG");
    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_ONLY");
}

TEST(ProductPcaSpectrum, DiagnosticOnlyStopsGenOptAbfsCalculationsExclusively)
{
    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_ONLY");
    EXPECT_FALSE(ABFs_Construct::PCA::product_pca_spectrum_should_stop_after_generation("gen_opt_abfs"));

    setenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_ONLY", "true", 1);
    EXPECT_TRUE(ABFs_Construct::PCA::product_pca_spectrum_should_stop_after_generation("gen_opt_abfs"));
    EXPECT_FALSE(ABFs_Construct::PCA::product_pca_spectrum_should_stop_after_generation("scf"));
    EXPECT_FALSE(ABFs_Construct::PCA::product_pca_spectrum_should_stop_after_generation("nscf"));
    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_ONLY");
}

TEST(ProductPcaSpectrum, UsesExplicitOutputPathOrStableDefault)
{
    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_FILE");
    EXPECT_EQ(ABFs_Construct::PCA::product_pca_spectrum_output_path(), "PRODUCT_PCA_SPECTRUM.dat");
    setenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_FILE", "/tmp/pca-spectrum.tsv", 1);
    EXPECT_EQ(ABFs_Construct::PCA::product_pca_spectrum_output_path(), "/tmp/pca-spectrum.tsv");
    unsetenv("ABACUS_PRODUCT_PCA_SPECTRUM_DIAG_FILE");
}

} // namespace
