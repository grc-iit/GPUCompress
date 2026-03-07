/**
 * @file xfer_tracker.cpp
 * @brief Global counters and summary dump for host↔device transfer tracking.
 */

#include "xfer_tracker.h"
#include <stdio.h>
#include <stdlib.h>

/* Check env var once at startup via static initializer */
static int xfer_check_env(void) {
    const char* val = getenv("GPUCOMPRESS_XFER_TRACK");
    return (val && val[0] == '1') ? 1 : 0;
}
int      g_xfer_enabled   = xfer_check_env();

int      g_xfer_h2d_count = 0;
int      g_xfer_d2h_count = 0;
int      g_xfer_d2d_count = 0;
int64_t  g_xfer_h2d_bytes = 0;
int64_t  g_xfer_d2h_bytes = 0;
int64_t  g_xfer_d2d_bytes = 0;
int      g_xfer_seq       = 0;

void xfer_tracker_dump(void) {
    if (!g_xfer_enabled) return;
    int total = g_xfer_h2d_count + g_xfer_d2h_count + g_xfer_d2d_count;
    int64_t total_bytes = g_xfer_h2d_bytes + g_xfer_d2h_bytes + g_xfer_d2d_bytes;
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║          cudaMemcpy TRANSFER SUMMARY             ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  H->D:  %4d calls   %12ld bytes          ║\n", g_xfer_h2d_count, (long)g_xfer_h2d_bytes);
    fprintf(stderr, "║  D->H:  %4d calls   %12ld bytes          ║\n", g_xfer_d2h_count, (long)g_xfer_d2h_bytes);
    fprintf(stderr, "║  D->D:  %4d calls   %12ld bytes          ║\n", g_xfer_d2d_count, (long)g_xfer_d2d_bytes);
    fprintf(stderr, "║  TOTAL: %4d calls   %12ld bytes          ║\n", total, (long)total_bytes);
    fprintf(stderr, "╚══════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
}

void xfer_tracker_reset(void) {
    g_xfer_h2d_count = 0;
    g_xfer_d2h_count = 0;
    g_xfer_d2d_count = 0;
    g_xfer_h2d_bytes = 0;
    g_xfer_d2h_bytes = 0;
    g_xfer_d2d_bytes = 0;
    g_xfer_seq       = 0;
}

void xfer_tracker_enable(int on) {
    g_xfer_enabled = on;
}
