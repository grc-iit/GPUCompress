#pragma once
/**
 * @file diagnostics_store.hpp
 * @brief Singleton that owns all per-chunk diagnostic history and cache stats.
 *
 * This is the single source of truth for runtime statistics collected by the
 * VOL connector.  The public C API in gpucompress.h delegates here.
 *
 * Callers inside the library (compress path, pool, VOL) use the singleton
 * directly rather than going through the C wrappers.
 */

#include <atomic>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "gpucompress.h"

namespace gpucompress {

class DiagnosticsStore {
public:
    /* Meyer's singleton — constructed on first use, destroyed at exit. */
    static DiagnosticsStore& instance() {
        static DiagnosticsStore s;
        return s;
    }

    DiagnosticsStore(const DiagnosticsStore&)            = delete;
    DiagnosticsStore& operator=(const DiagnosticsStore&) = delete;

    /* ── Chunk history ─────────────────────────────────────────────── */

    /**
     * Append one diagnostic record.  Returns the index assigned to this
     * chunk (used to back-fill decompression timing later), or -1 on OOM.
     */
    int record(const gpucompress_chunk_diag_t& entry) {
        std::lock_guard<std::mutex> lk(mutex_);
        int idx = count_.fetch_add(1);
        if (idx >= cap_) {
            int new_cap = (cap_ == 0) ? 4096 : cap_ * 2;
            auto* p = static_cast<gpucompress_chunk_diag_t*>(
                realloc(history_, (size_t)new_cap * sizeof(gpucompress_chunk_diag_t)));
            if (!p) { count_.fetch_sub(1); return -1; }
            history_ = p;
            cap_     = new_cap;
        }
        history_[idx] = entry;
        return idx;
    }

    /** Zero-fill a slot and return its index (for incremental filling). */
    int allocSlot() {
        std::lock_guard<std::mutex> lk(mutex_);
        int idx = count_.fetch_add(1);
        if (idx >= cap_) {
            int new_cap = (cap_ == 0) ? 4096 : cap_ * 2;
            auto* p = static_cast<gpucompress_chunk_diag_t*>(
                realloc(history_, (size_t)new_cap * sizeof(gpucompress_chunk_diag_t)));
            if (!p) { count_.fetch_sub(1); return -1; }
            history_ = p;
            cap_     = new_cap;
        }
        memset(&history_[idx], 0, sizeof(gpucompress_chunk_diag_t));
        return idx;
    }

    int count() const { return count_.load(); }

    int getDiag(int idx, gpucompress_chunk_diag_t* out) const {
        if (idx < 0 || !out) return -1;
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx >= count_.load() || idx >= cap_) return -1;
        *out = history_[idx];
        return 0;
    }

    /** Back-fill actual decompression timing (called from VOL read path). */
    void recordDecompMs(int idx, float ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        gpucompress_chunk_diag_t& h = history_[idx];
        float clamped = std::max(1.0f, ms);
        h.decompression_ms     = clamped;
        h.decompression_ms_raw = ms;
        float pred_dt = std::max(1.0f, h.predicted_decomp_time);
        if (pred_dt > 0.0f)
            h.decomp_time_mape = std::abs(clamped - pred_dt) / clamped;
    }

    /** Back-fill VOL Stage 2 pipeline timing (pool acquire, D→H, I/O wait). */
    void recordVolTiming(int idx, float pool_acquire_ms,
                         float d2h_copy_ms, float io_queue_wait_ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        history_[idx].vol_pool_acquire_ms  = pool_acquire_ms;
        history_[idx].vol_d2h_copy_ms      = d2h_copy_ms;
        history_[idx].vol_io_queue_wait_ms = io_queue_wait_ms;
    }

    /** Back-fill VOL Stage 1 pipeline timing (stats alloc, stats copy, WQ wait). */
    void recordS1Timing(int idx, float stats_malloc_ms,
                        float stats_copy_ms, float wq_post_wait_ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        history_[idx].vol_stats_malloc_ms  = stats_malloc_ms;
        history_[idx].vol_stats_copy_ms    = stats_copy_ms;
        history_[idx].vol_wq_post_wait_ms  = wq_post_wait_ms;
    }

    /** Set diag_record_ms on an existing slot (called after record() returns). */
    void setDiagRecordMs(int idx, float ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        history_[idx].diag_record_ms = ms;
    }

    /** Reset the history (call before each H5Dwrite). Does not free memory. */
    void reset() { count_.store(0); }

    /* ── Cache stats ───────────────────────────────────────────────── */

    void incrementCacheHit()  { cache_hits_.fetch_add(1);   }
    void incrementCacheMiss() { cache_misses_.fetch_add(1); }

    void getCacheStats(int* hits, int* misses) const {
        if (hits)   *hits   = cache_hits_.load();
        if (misses) *misses = cache_misses_.load();
    }

    void resetCacheStats() {
        cache_hits_.store(0);
        cache_misses_.store(0);
    }

private:
    DiagnosticsStore()  = default;
    ~DiagnosticsStore() { free(history_); }

    mutable std::mutex           mutex_;
    gpucompress_chunk_diag_t*    history_      = nullptr;
    int                          cap_          = 0;
    std::atomic<int>             count_{0};
    std::atomic<int>             cache_hits_{0};
    std::atomic<int>             cache_misses_{0};
};

} // namespace gpucompress
