/**
 * calloc_fault.c — LD_PRELOAD interposer for fault injection.
 *
 * Build:  gcc -shared -fPIC -o calloc_fault.so calloc_fault.c -ldl
 * Usage:  CALLOC_FAIL_AFTER=50 LD_PRELOAD=./calloc_fault.so ./test_h7_null_calloc
 *
 * After CALLOC_FAIL_AFTER successful calloc calls, returns NULL once,
 * then resumes normal operation. This targets the new_obj() path in
 * H5VLgpucompress.cu which doesn't check for NULL.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int g_fail_after  = -1;   /* -1 = not initialized */
static int g_call_count  = 0;
static int g_faulted     = 0;

typedef void* (*real_calloc_t)(size_t, size_t);

static real_calloc_t get_real_calloc(void) {
    static real_calloc_t real_fn = NULL;
    if (!real_fn) {
        real_fn = (real_calloc_t)dlsym(RTLD_NEXT, "calloc");
    }
    return real_fn;
}

void *calloc(size_t nmemb, size_t size) {
    real_calloc_t real_calloc = get_real_calloc();
    if (!real_calloc) return NULL;

    /* Lazy init from env */
    if (g_fail_after < 0) {
        const char *env = getenv("CALLOC_FAIL_AFTER");
        g_fail_after = env ? atoi(env) : 0;
        if (g_fail_after <= 0) g_fail_after = 999999999; /* effectively disabled */
    }

    g_call_count++;

    /* Fail once at the threshold */
    if (!g_faulted && g_call_count > g_fail_after) {
        g_faulted = 1;
        fprintf(stderr, "[calloc_fault] Injecting NULL at call #%d "
                "(nmemb=%zu, size=%zu)\n", g_call_count, nmemb, size);
        return NULL;
    }

    return real_calloc(nmemb, size);
}
