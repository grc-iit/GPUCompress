/**
 * @file xfer_tracker.h
 * @brief Host↔Device transfer tracker for diagnosing unnecessary cudaMemcpy calls.
 *
 * Usage: include this header then call XFER_TRACK(...) before every cudaMemcpy/cudaMemcpyAsync.
 * At program exit or on demand, call xfer_tracker_dump() to print a summary.
 */

#ifndef XFER_TRACKER_H
#define XFER_TRACKER_H

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime on/off — default OFF.
 * Enable via:  xfer_tracker_enable(1)   or   env GPUCOMPRESS_XFER_TRACK=1  */
extern int      g_xfer_enabled;

/* Global counters — defined in xfer_tracker.cpp, zero-initialized */
extern int      g_xfer_h2d_count;
extern int      g_xfer_d2h_count;
extern int      g_xfer_d2d_count;
extern int64_t  g_xfer_h2d_bytes;
extern int64_t  g_xfer_d2h_bytes;
extern int64_t  g_xfer_d2d_bytes;
extern int      g_xfer_seq;          /* monotonic sequence number */

static inline const char* xfer_kind_str(cudaMemcpyKind k) {
    switch (k) {
        case cudaMemcpyHostToDevice:   return "H->D";
        case cudaMemcpyDeviceToHost:   return "D->H";
        case cudaMemcpyDeviceToDevice: return "D->D";
        default:                       return "????";
    }
}

static inline void xfer_track(const char* file, int line, const char* label,
                               size_t bytes, cudaMemcpyKind kind) {
    if (!g_xfer_enabled) return;
    int seq = __sync_fetch_and_add(&g_xfer_seq, 1);
    switch (kind) {
        case cudaMemcpyHostToDevice:
            __sync_fetch_and_add(&g_xfer_h2d_count, 1);
            __sync_fetch_and_add(&g_xfer_h2d_bytes, (int64_t)bytes);
            break;
        case cudaMemcpyDeviceToHost:
            __sync_fetch_and_add(&g_xfer_d2h_count, 1);
            __sync_fetch_and_add(&g_xfer_d2h_bytes, (int64_t)bytes);
            break;
        case cudaMemcpyDeviceToDevice:
            __sync_fetch_and_add(&g_xfer_d2d_count, 1);
            __sync_fetch_and_add(&g_xfer_d2d_bytes, (int64_t)bytes);
            break;
        default: break;
    }
    fprintf(stderr, "[XFER #%03d %s] %s:%d  %s  %zu bytes\n",
            seq, xfer_kind_str(kind), file, line, label, bytes);
}

void xfer_tracker_dump(void);
void xfer_tracker_reset(void);

/** Enable (on=1) or disable (on=0) transfer tracking at runtime. */
void xfer_tracker_enable(int on);

#ifdef __cplusplus
}
#endif

/**
 * Macro: place before every cudaMemcpy / cudaMemcpyAsync call.
 *   XFER_TRACK("compress: input data", nbytes, cudaMemcpyHostToDevice);
 */
#define XFER_TRACK(label, bytes, kind) \
    xfer_track(__FILE__, __LINE__, (label), (size_t)(bytes), (kind))

#endif /* XFER_TRACKER_H */
