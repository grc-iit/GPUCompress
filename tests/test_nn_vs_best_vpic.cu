/**
 * Test NN accuracy on synthetic VPIC-like data patterns.
 *
 * VPIC Harris sheet produces 16 interleaved float fields per cell:
 *   ex, ey, ez, div_e_err, bx, by, bz, div_b_err,
 *   tcax, tcay, tcaz, rhob, jfx, jfy, jfz, rhof
 *
 * After ~100 warmup steps, the physics develops:
 *   - Bz: smooth tanh(x/L) profile → very compressible
 *   - Ex,Ey,Ez: noisy particle-induced fields → low compressibility
 *   - Ghost cells (boundaries): mostly zeros → extremely compressible
 *   - Current densities: spatially varying, moderate structure
 *
 * Different chunks of the flat 1D array see different mixes of these
 * 16 variables, explaining the dramatic ratio variation (1.2x to 1500x).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

static const char* algo_names[] = {
    "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

struct Pattern {
    const char* name;
    const char* description;
    void (*gen)(float*, size_t);
};

// ── VPIC-like pattern generators ──────────────────────────────────

// 1. Ghost cell region: almost entirely zeros (boundary padding)
static void gen_ghost_cells(float* d, size_t n) {
    memset(d, 0, n * sizeof(float));
    // Occasional tiny leakage from boundary conditions
    for (size_t i = 0; i < n; i += 256)
        d[i] = 1e-15f;
}

// 2. Bz field: tanh profile (Harris sheet magnetic field)
//    Bz(x) = b0 * tanh(x / L), stored in every 16th position
static void gen_bz_tanh(float* d, size_t n) {
    double b0 = 0.01;   // typical b0 in normalized units
    double L  = 10.0;    // current sheet half-width
    double Lx = 800.0;   // domain length
    for (size_t i = 0; i < n; i++) {
        // Simulate interleaved: every 16th float is Bz
        if (i % 16 == 6) {  // bz is field index 6
            double x = -0.5*Lx + Lx * ((double)(i/16) / (double)(n/16));
            d[i] = (float)(b0 * tanh(x / L));
        } else {
            d[i] = 0.0f;  // other fields zero in this chunk
        }
    }
}

// 3. Pure Bz field (non-interleaved, like a chunk that's all Bz)
static void gen_pure_tanh(float* d, size_t n) {
    double b0 = 0.01;
    double L  = 10.0;
    double Lx = 800.0;
    for (size_t i = 0; i < n; i++) {
        double x = -0.5*Lx + Lx * ((double)i / (double)n);
        d[i] = (float)(b0 * tanh(x / L));
    }
}

// 4. Electric field (noisy, particle-induced)
//    After warmup, E-fields have thermal noise + coherent structures
static void gen_e_field_noisy(float* d, size_t n) {
    srand(42);
    double thermal = 1e-4;
    for (size_t i = 0; i < n; i++) {
        // Base: small coherent wave
        double coherent = 1e-3 * sin((double)i * 0.02);
        // Noise: thermal fluctuations
        double noise = thermal * ((double)rand()/RAND_MAX - 0.5);
        d[i] = (float)(coherent + noise);
    }
}

// 5. Current density (Jfy): peaked near current sheet, zero far away
static void gen_current_density(float* d, size_t n) {
    double L = 10.0;
    double Lx = 800.0;
    for (size_t i = 0; i < n; i++) {
        double x = -0.5*Lx + Lx * ((double)i / (double)n);
        // J = b0/(mu0*L) * sech^2(x/L)
        double sech = 1.0 / cosh(x / L);
        d[i] = (float)(0.001 * sech * sech);
    }
}

// 6. Mixed interleaved: realistic 16-field chunk from mid-domain
//    Contains all field types interleaved
static void gen_mixed_fields_mid(float* d, size_t n) {
    srand(123);
    double b0 = 0.01, L = 10.0, Lx = 800.0;
    for (size_t i = 0; i < n; i++) {
        int field = i % 16;
        double cell = (double)(i / 16) / (double)(n / 16);
        double x = -0.5*Lx + Lx * cell;
        double sech = 1.0 / cosh(x / L);
        switch (field) {
            case 0: // ex - noisy
            case 1: // ey - noisy
            case 2: // ez - noisy
                d[i] = (float)(1e-4 * ((double)rand()/RAND_MAX - 0.5));
                break;
            case 3: // div_e_err - near zero
                d[i] = (float)(1e-12 * ((double)rand()/RAND_MAX - 0.5));
                break;
            case 4: // bx - small perturbation
                d[i] = (float)(1e-5 * sin(cell * 6.28));
                break;
            case 5: // by - zero initially
                d[i] = 0.0f;
                break;
            case 6: // bz - tanh profile (dominant)
                d[i] = (float)(b0 * tanh(x / L));
                break;
            case 7: // div_b_err - near zero
                d[i] = (float)(1e-14 * ((double)rand()/RAND_MAX - 0.5));
                break;
            case 8:  // tcax
            case 9:  // tcay
            case 10: // tcaz - temporary accumulators, often zero
                d[i] = 0.0f;
                break;
            case 11: // rhob - bound charge, near zero
                d[i] = (float)(1e-10 * sech * sech);
                break;
            case 12: // jfx - small
                d[i] = (float)(1e-5 * ((double)rand()/RAND_MAX - 0.5));
                break;
            case 13: // jfy - peaked at current sheet
                d[i] = (float)(0.001 * sech * sech);
                break;
            case 14: // jfz - small
                d[i] = (float)(1e-5 * ((double)rand()/RAND_MAX - 0.5));
                break;
            case 15: // rhof - free charge
                d[i] = (float)(1e-6 * sech * sech);
                break;
        }
    }
}

// 7. Edge region: ghost cells + some real data (boundary chunk)
static void gen_boundary_chunk(float* d, size_t n) {
    // First 3/4 is ghost (zeros), last 1/4 has real field data
    size_t boundary = 3 * n / 4;
    memset(d, 0, boundary * sizeof(float));
    srand(77);
    for (size_t i = boundary; i < n; i++) {
        int field = i % 16;
        if (field == 6) // bz at boundary
            d[i] = 0.01f; // tanh(large x) ≈ b0
        else if (field < 3) // e-fields
            d[i] = (float)(1e-4 * ((double)rand()/RAND_MAX - 0.5));
        else
            d[i] = 0.0f;
    }
}

// 8. Reconnection region: turbulent mix (center of current sheet)
//    After warmup, the reconnection zone has complex, noisy fields
static void gen_reconnection_zone(float* d, size_t n) {
    srand(314);
    for (size_t i = 0; i < n; i++) {
        int field = i % 16;
        double cell = (double)(i / 16) / (double)(n / 16);
        switch (field) {
            case 0: case 1: case 2: // E-fields: strong fluctuations
                d[i] = (float)(0.01 * ((double)rand()/RAND_MAX - 0.5));
                break;
            case 4: case 5: case 6: // B-fields: large + noise
                d[i] = (float)(0.01 * sin(cell * 3.14) +
                        0.005 * ((double)rand()/RAND_MAX - 0.5));
                break;
            case 12: case 13: case 14: // currents: strong
                d[i] = (float)(0.005 * ((double)rand()/RAND_MAX - 0.5));
                break;
            default: // corrections, accumulators: small noise
                d[i] = (float)(1e-8 * ((double)rand()/RAND_MAX - 0.5));
                break;
        }
    }
}

// 9. Smooth E-field region (far from reconnection, coherent waves)
static void gen_smooth_efield(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double cell = (double)i / (double)n;
        d[i] = (float)(1e-3 * sin(cell * 20.0) * cos(cell * 3.0));
    }
}

// 10. Density structure: n(x) = n0 * sech^2(x/L) — Harris equilibrium
static void gen_density_sech2(float* d, size_t n) {
    double L = 10.0, Lx = 800.0;
    for (size_t i = 0; i < n; i++) {
        double x = -0.5*Lx + Lx * ((double)i / (double)n);
        double sech = 1.0 / cosh(x / L);
        d[i] = (float)(sech * sech);
    }
}

// 11. Alternating regimes: simulates chunk boundary crossing
//     First half = ghost cells (zeros), second half = noisy E-field
static void gen_regime_transition(float* d, size_t n) {
    srand(99);
    size_t mid = n / 2;
    memset(d, 0, mid * sizeof(float));
    for (size_t i = mid; i < n; i++)
        d[i] = (float)(0.01 * ((double)rand()/RAND_MAX - 0.5));
}

// 12. Near-constant with tiny perturbations (By field, nearly zero)
static void gen_near_constant_perturbed(float* d, size_t n) {
    srand(55);
    for (size_t i = 0; i < n; i++)
        d[i] = (float)(1e-10 * ((double)rand()/RAND_MAX - 0.5));
}

// 13. Particle velocities: Maxwellian distribution
static void gen_maxwellian_velocity(float* d, size_t n) {
    srand(200);
    double uthe = 0.1; // thermal velocity
    for (size_t i = 0; i < n; i++) {
        // Box-Muller for Gaussian
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
        double u2 = (double)rand() / (double)RAND_MAX;
        d[i] = (float)(uthe * sqrt(-2.0 * log(u1)) * cos(6.2832 * u2));
    }
}

// 14. Particle positions: uniform in [0, Ly]
static void gen_uniform_positions(float* d, size_t n) {
    srand(300);
    double Ly = 800.0;
    for (size_t i = 0; i < n; i++)
        d[i] = (float)(Ly * (double)rand() / (double)RAND_MAX);
}

static const char* find_weights() {
    static char buf[512];
    snprintf(buf, sizeof(buf), "%s/GPUCompress/neural_net/weights/model.nnwt",
             getenv("HOME") ? getenv("HOME") : ".");
    FILE* f = fopen(buf, "rb");
    if (f) { fclose(f); return buf; }
    f = fopen("neural_net/weights/model.nnwt", "rb");
    if (f) { fclose(f); return "neural_net/weights/model.nnwt"; }
    return NULL;
}

int main() {
    const char* w = find_weights();
    if (!w) { fprintf(stderr, "No weights file found\n"); return 1; }

    gpucompress_error_t err = gpucompress_init(w);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }

    Pattern patterns[] = {
        {"ghost_cells",     "Boundary padding, ~all zeros",         gen_ghost_cells},
        {"bz_tanh_intrlv",  "Bz=b0*tanh(x/L) interleaved/16",     gen_bz_tanh},
        {"pure_tanh",       "Pure tanh profile (Bz only)",          gen_pure_tanh},
        {"e_field_noisy",   "Particle-induced E-field + noise",     gen_e_field_noisy},
        {"current_density", "Jfy = sech^2(x/L) peaked sheet",      gen_current_density},
        {"mixed_16fields",  "All 16 fields interleaved, mid-domain",gen_mixed_fields_mid},
        {"boundary_chunk",  "3/4 ghost + 1/4 real data",           gen_boundary_chunk},
        {"reconnect_zone",  "Turbulent reconnection center",        gen_reconnection_zone},
        {"smooth_efield",   "Coherent wave E-field",                gen_smooth_efield},
        {"density_sech2",   "Harris density: n0*sech^2(x/L)",       gen_density_sech2},
        {"regime_trans",    "Half zeros, half noisy E-field",       gen_regime_transition},
        {"near_const_pert", "By ≈ 0 with 1e-10 perturbation",      gen_near_constant_perturbed},
        {"maxwellian_vel",  "Particle velocities (Gaussian)",       gen_maxwellian_velocity},
        {"uniform_pos",     "Particle positions (uniform)",         gen_uniform_positions},
    };
    int np = sizeof(patterns) / sizeof(patterns[0]);

    size_t sizes[] = {65536, 1<<20, 4*1024*1024};
    const char* sz_names[] = {"64KB", "1MB", "4MB"};

    for (int si = 0; si < 3; si++) {
        size_t dsz = sizes[si];
        size_t n = dsz / sizeof(float);
        float* data = (float*)malloc(dsz);
        size_t max_out = gpucompress_max_compressed_size(dsz);
        void* output = malloc(max_out);

        printf("\n");
        printf("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║  VPIC-LIKE DATA — CHUNK SIZE: %-6s                                                                         ║\n", sz_names[si]);
        printf("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ %-18s │ %7s %7s %7s %7s %7s %7s %7s %7s │ %-10s │ %-10s │ %-5s ║\n",
               "Pattern", "lz4", "snappy", "deflat", "gdefl", "zstd", "ans", "cascad", "bitcomp",
               "BEST", "NN_PICK", "MATCH");
        printf("╠════════════════════╪═════════════════════════════════════════════════════════════════╪════════════╪════════════╪═══════╣\n");

        int correct = 0, close_count = 0, total = 0;

        for (int p = 0; p < np; p++) {
            patterns[p].gen(data, n);

            // Run all 8 algorithms with shuffle
            double ratios[8] = {0};
            for (int a = 0; a < 8; a++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)(a + 1);
                cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
                gpucompress_stats_t stats;
                size_t sz = max_out;
                err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);
                ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
            }

            int best = 0;
            for (int a = 1; a < 8; a++) if (ratios[a] > ratios[best]) best = a;

            // Ask NN
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
            cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
            gpucompress_stats_t stats;
            size_t sz = max_out;
            err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);

            int nn_algo_idx = -1;
            const char* nn_algo = "???";
            if (err == GPUCOMPRESS_SUCCESS) {
                nn_algo_idx = stats.nn_final_action % 8;
                nn_algo = algo_names[nn_algo_idx];
            }

            int match = (nn_algo_idx == best);
            int close = 0;
            if (nn_algo_idx >= 0 && ratios[best] > 0)
                close = (ratios[nn_algo_idx] >= ratios[best] * 0.95);
            if (err == GPUCOMPRESS_SUCCESS) {
                total++;
                if (match) correct++;
                else if (close) close_count++;
            }

            printf("║ %-18s │", patterns[p].name);
            for (int a = 0; a < 8; a++) {
                if (a == best)
                    printf(" %6.1f*", ratios[a]);
                else
                    printf(" %7.1f", ratios[a]);
            }
            printf(" │ %-10s │ %-10s │ %s ║\n",
                   algo_names[best], nn_algo,
                   match ? "YES" : (close ? "~95%" : "NO"));
        }

        printf("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ Exact: %d/%d (%.0f%%)  Within 5%%: %d/%d  Total good: %d/%d (%.0f%%)                                                    ║\n",
               correct, total, total > 0 ? 100.0*correct/total : 0.0,
               close_count, total,
               correct + close_count, total,
               total > 0 ? 100.0*(correct+close_count)/total : 0.0);
        printf("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

        free(data);
        free(output);
    }

    gpucompress_cleanup();
    return 0;
}
