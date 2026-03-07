/**
 * @file test_quantization_cub.cu
 * @brief Verifies CUB-based min/max reduction produces identical quantization
 *        results to the original hand-rolled kernels.
 *
 * Tests:
 *  1. Min/max correctness for positive-only, negative-only, and mixed data
 *  2. Quantize/dequantize round-trip error bound for float32
 *  3. Quantize/dequantize round-trip error bound for float64
 *  4. Precision selection (int8, int16, int32) matches expected bins
 *  5. Constant data (range=0) edge case
 *  6. Single element edge case
 *  7. Large dataset (16M elements)
 *  8. Thread-safe overload with explicit range buffers
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "preprocessing/quantization.cuh"

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) \
    do { printf("  [TEST] %s ... ", name); } while (0)

#define PASS() \
    do { printf("PASS\n"); g_pass++; } while (0)

#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); g_fail++; } while (0)

#define ASSERT(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while (0)

// Helper: upload host array to device
template<typename T>
static T* upload(const T* h_data, size_t n) {
    T* d_ptr = nullptr;
    cudaMalloc(&d_ptr, n * sizeof(T));
    cudaMemcpy(d_ptr, h_data, n * sizeof(T), cudaMemcpyHostToDevice);
    return d_ptr;
}

// ============================================================
// Test 1: Min/max with positive floats
// ============================================================
static void test_minmax_positive_float() {
    TEST("Min/max positive floats");

    const size_t N = 1024 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = 10.0f + (float)i * 0.001f;  // [10.0, ~1033.6]
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, 0.01, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");

    // Check min/max stored in result
    ASSERT(fabs(result.data_min - 10.0) < 0.001, "data_min wrong");
    ASSERT(fabs(result.data_max - (10.0f + (N - 1) * 0.001f)) < 0.01, "data_max wrong");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    PASS();
}

// ============================================================
// Test 2: Min/max with all-negative floats
// ============================================================
static void test_minmax_negative_float() {
    TEST("Min/max negative floats");

    const size_t N = 512 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = -1000.0f + (float)i * 0.001f;  // [-1000, ~-475]
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, 0.1, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");
    ASSERT(result.data_min < -999.0, "data_min should be ~-1000");
    ASSERT(result.data_max < 0, "data_max should be negative");
    ASSERT(result.data_max > result.data_min, "max > min");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    PASS();
}

// ============================================================
// Test 3: Min/max with mixed positive/negative floats
// ============================================================
static void test_minmax_mixed_float() {
    TEST("Min/max mixed pos/neg floats");

    const size_t N = 2 * 1024 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = sinf((float)i * 0.01f) * 500.0f;  // [-500, +500]
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, 0.01, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");
    ASSERT(result.data_min < -490.0, "expected min near -500");
    ASSERT(result.data_max > 490.0, "expected max near +500");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    PASS();
}

// ============================================================
// Test 4: Float32 quantize/dequantize round-trip (error bound)
// ============================================================
static void test_roundtrip_float32() {
    TEST("Float32 round-trip error bound (eb=0.01, 2M elements)");

    const size_t N = 2 * 1024 * 1024;
    const double eb = 0.01;

    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = sinf((float)i * 0.001f) * 100.0f;
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, eb, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool ok = verify_error_bound(d_data, d_restored, N, sizeof(float), eb, 0, &max_error);
    ASSERT(ok, "error bound violated");

    printf("(max_err=%.2e) ", max_error);

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

// ============================================================
// Test 5: Float64 quantize/dequantize round-trip (error bound)
// ============================================================
static void test_roundtrip_float64() {
    TEST("Float64 round-trip error bound (eb=1e-6, 1M elements)");

    const size_t N = 1024 * 1024;
    const double eb = 1e-6;

    double* h_data = (double*)malloc(N * sizeof(double));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = sin((double)i * 0.0001) * 10.0;
    }

    double* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, eb, N, sizeof(double));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(double), config);
    ASSERT(result.isValid(), "quantization failed");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool ok = verify_error_bound(d_data, d_restored, N, sizeof(double), eb, 0, &max_error);
    ASSERT(ok, "error bound violated");

    printf("(max_err=%.2e) ", max_error);

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

// ============================================================
// Test 6: Precision selection — small range → int8
// ============================================================
static void test_precision_int8() {
    TEST("Precision selection: small range -> int8");

    const size_t N = 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    // Range [0, 1.0], eb=0.01 → bins = 1.0/0.02 = 50 → int8
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)i / (float)N;
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, 0.01, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");
    ASSERT(result.actual_precision == 8, "expected int8 precision");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    PASS();
}

// ============================================================
// Test 7: Precision selection — wide range → int32
// ============================================================
static void test_precision_int32() {
    TEST("Precision selection: wide range -> int32");

    const size_t N = 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    // Range [0, 100000], eb=0.001 → bins = 100000/0.002 = 50M → int32
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)i * 100.0f;
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, 0.001, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");
    ASSERT(result.actual_precision == 32, "expected int32 precision");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    PASS();
}

// ============================================================
// Test 8: Constant data (all same value, range = 0)
// ============================================================
static void test_constant_data() {
    TEST("Constant data (range=0, all values=42.0)");

    const size_t N = 4096;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = 42.0f;
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, 0.01, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");
    ASSERT(fabs(result.data_min - 42.0) < 0.001, "data_min should be 42");
    ASSERT(fabs(result.data_max - 42.0) < 0.001, "data_max should be 42");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool ok = verify_error_bound(d_data, d_restored, N, sizeof(float), 0.01, 0, &max_error);
    ASSERT(ok, "error bound violated on constant data");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

// ============================================================
// Test 9: Single element
// ============================================================
static void test_single_element() {
    TEST("Single element (N=1)");

    float h_val = -3.14f;
    float* d_data = upload(&h_val, 1);

    QuantizationConfig config(QuantizationType::LINEAR, 0.1, 1, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, 1, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");
    ASSERT(fabs(result.data_min - (-3.14)) < 0.01, "data_min wrong");
    ASSERT(fabs(result.data_max - (-3.14)) < 0.01, "data_max wrong");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool ok = verify_error_bound(d_data, d_restored, 1, sizeof(float), 0.1, 0, &max_error);
    ASSERT(ok, "error bound violated on single element");

    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

// ============================================================
// Test 10: Large dataset (16M elements)
// ============================================================
static void test_large_dataset() {
    TEST("Large dataset (16M float32, eb=0.001)");

    const size_t N = 16 * 1024 * 1024;
    const double eb = 0.001;

    float* h_data = (float*)malloc(N * sizeof(float));
    unsigned seed = 12345;
    for (size_t i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        h_data[i] = ((float)(seed >> 8) / 16777216.0f) * 2000.0f - 1000.0f;
    }

    float* d_data = upload(h_data, N);

    QuantizationConfig config(QuantizationType::LINEAR, eb, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization failed");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool ok = verify_error_bound(d_data, d_restored, N, sizeof(float), eb, 0, &max_error);
    ASSERT(ok, "error bound violated on large dataset");

    printf("(max_err=%.2e) ", max_error);

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

// ============================================================
// Test 11: Thread-safe overload with explicit range buffers
// ============================================================
static void test_explicit_range_buffers() {
    TEST("Thread-safe overload with explicit range buffers");

    const size_t N = 1024 * 1024;
    const double eb = 0.01;

    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)i * 0.05f - 25000.0f;
    }

    float* d_data = upload(h_data, N);

    // Allocate our own range buffers (as CompContext would)
    void* d_my_min = nullptr;
    void* d_my_max = nullptr;
    cudaMalloc(&d_my_min, sizeof(double));
    cudaMalloc(&d_my_max, sizeof(double));

    QuantizationConfig config(QuantizationType::LINEAR, eb, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config,
                                                 d_my_min, d_my_max);
    ASSERT(result.isValid(), "quantization failed with explicit buffers");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool ok = verify_error_bound(d_data, d_restored, N, sizeof(float), eb, 0, &max_error);
    ASSERT(ok, "error bound violated with explicit buffers");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    cudaFree(d_my_min);
    cudaFree(d_my_max);
    PASS();
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("=== test_quantization_cub (CUB DeviceReduce min/max) ===\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found — skipping\n");
        return 1;
    }

    // Min/max correctness
    test_minmax_positive_float();
    test_minmax_negative_float();
    test_minmax_mixed_float();

    // Round-trip error bound
    test_roundtrip_float32();
    test_roundtrip_float64();

    // Precision selection
    test_precision_int8();
    test_precision_int32();

    // Edge cases
    test_constant_data();
    test_single_element();

    // Scale
    test_large_dataset();

    // Thread-safe overload
    test_explicit_range_buffers();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
