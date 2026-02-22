#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include "gpucompress.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.bin> [weights.nnwt]\n", argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    const char* weights = (argc >= 3) ? argv[2] : nullptr;

    // Read file
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    if (!ifs) { fprintf(stderr, "Cannot open %s\n", filepath); return 1; }
    size_t file_size = ifs.tellg();
    ifs.seekg(0);
    std::vector<char> raw(file_size);
    ifs.read(raw.data(), file_size);
    ifs.close();

    printf("Input: %s (%zu bytes)\n", filepath, file_size);

    // Init
    gpucompress_init(nullptr);
    if (weights) gpucompress_load_nn(weights);

    // Compress
    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = weights ? GPUCOMPRESS_ALGO_AUTO : GPUCOMPRESS_ALGO_LZ4;
    cfg.error_bound = 0.0;

    size_t max_out = gpucompress_max_compressed_size(file_size);
    std::vector<char> comp(max_out);
    size_t comp_size = max_out;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    int rc = gpucompress_compress(raw.data(), file_size, comp.data(), &comp_size, &cfg, &stats);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "Compress failed: %s\n", gpucompress_error_string((gpucompress_error_t)rc));
        gpucompress_cleanup();
        return 1;
    }

    printf("Compressed: %zu -> %zu bytes (ratio=%.3f)\n", file_size, comp_size, stats.compression_ratio);
    printf("Algorithm:  %s\n", gpucompress_algorithm_name(stats.algorithm_used));
    printf("Entropy:    %.3f bits\n", stats.entropy_bits);
    if (weights) {
        printf("Predicted ratio: %.3f\n", stats.predicted_ratio);
        printf("Predicted CT:    %.3f ms\n", stats.predicted_comp_time_ms);
    }

    // Decompress
    std::vector<char> decomp(file_size);
    size_t decomp_size = file_size;
    rc = gpucompress_decompress(comp.data(), comp_size, decomp.data(), &decomp_size);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "Decompress failed: %s\n", gpucompress_error_string((gpucompress_error_t)rc));
        gpucompress_cleanup();
        return 1;
    }

    // Verify
    if (decomp_size == file_size && memcmp(raw.data(), decomp.data(), file_size) == 0) {
        printf("Roundtrip:  OK (lossless)\n");
    } else {
        printf("Roundtrip:  MISMATCH (decompressed %zu bytes)\n", decomp_size);
    }

    gpucompress_cleanup();
    return 0;
}
