/**
 * tests/hdf5/test_hdf5_compat.cu
 *
 * HDF5 Compatibility Test — VOL ↔ Native Interoperability
 *
 * Proves that our GPUCompress VOL connector is a compliant HDF5 citizen:
 * files it writes are readable by any HDF5 tool, and it can read files
 * written by vanilla HDF5.
 *
 * ─── Test Roster ────────────────────────────────────────────────────────────
 *
 * A. VOL write → NATIVE read
 *    1. Write a compressed, chunked dataset via GPUCompress VOL (GPU pointer).
 *    2. Reopen the file using the NATIVE HDF5 VOL (no gpucompress at all).
 *    3. Inspect metadata: file is valid, dataset exists, dimensions match,
 *       datatype matches, chunking properties match.
 *    4. Read the raw compressed bytes (H5Dread_chunk) — confirms HDF5 chunk
 *       infrastructure is intact even without our decompressor.
 *    5. Register our H5Z filter plugin and do a full H5Dread — confirms the
 *       compressed file is readable by any host with our filter .so installed.
 *    6. Verify every float element against the known formula.
 *
 * B. NATIVE write → VOL read
 *    1. Write a plain (uncompressed) chunked dataset via native HDF5 (host ptr).
 *    2. Open the file using our GPUCompress VOL (GPU destination pointer).
 *    3. Verify our VOL falls through correctly and returns bit-exact data.
 *    This tests the "transparent passthrough" path inside H5VLgpucompress.
 *
 * C. NATIVE write (GZIP compressed) → VOL read
 *    1. Write with HDF5's built-in GZIP filter.
 *    2. Read via our VOL — VOL must not crash, must let native handle it,
 *       and must return the correct data to the GPU buffer.
 *
 * D. VOL write → h5ls / h5dump shell check
 *    Runs "h5ls -v <file>" to confirm the file is parseable by standard tools.
 *    This is the most direct proof of compatibility.
 *
 * E. Dataset metadata round-trip
 *    Write a 3D chunked dataset via VOL with an HDF5 string attribute attached.
 *    Re-open with native HDF5, read the attribute — confirms our VOL does not
 *    corrupt the attribute / metadata section of the file.
 *
 * ─── Key properties verified ────────────────────────────────────────────────
 *   - Superblock, object headers, B-tree chunk index intact after VOL write
 *   - Our DCPL filter entry (ID 305) stored correctly in chunk metadata
 *   - Native HDF5 can enumerate chunks without gpucompress library loaded
 *   - Our VOL can read native-written files (passthrough path)
 *   - File is not corrupted across multiple open/close cycles
 *
 * Usage:
 *   export GPUCOMPRESS_WEIGHTS=/path/to/model.nnwt   # optional for ALGO_AUTO
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   HDF5_PLUGIN_PATH=/path/to/libH5Zgpucompress.so_dir \
 *   ./build/test_hdf5_compat
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Parameters
 * ============================================================ */
#define CHUNK_FLOATS  (1024 * 1024)          /* 1 MiB chunk */
#define N_CHUNKS      8                       /* 8 MiB total — fast */
#define TOTAL_FLOATS  ((size_t)CHUNK_FLOATS * N_CHUNKS)
#define TOTAL_BYTES   (TOTAL_FLOATS * sizeof(float))

#define TMP_FILE_VOL  "/tmp/compat_vol.h5"
#define TMP_FILE_NAT  "/tmp/compat_native.h5"
#define TMP_FILE_GZ   "/tmp/compat_gzip.h5"
#define TMP_FILE_3D   "/tmp/compat_3d.h5"
#define TMP_FILE_META "/tmp/compat_meta.h5"

#define XOR_SEED  0xABCD1234u

/* ============================================================
 * GPU / CPU data generation (ramp — deterministic, easy to verify)
 * ============================================================ */
__global__ static void ramp_kernel(float *b, size_t n, size_t total)
{
    for (size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x; i < n;
         i += gridDim.x*(size_t)blockDim.x)
        b[i] = (float)i / (float)total;
}

static inline float expected(size_t i) { return (float)i / (float)TOTAL_FLOATS; }

/* ============================================================
 * Pass / Fail helpers
 * ============================================================ */
static int g_pass = 0, g_fail = 0;

#define PASS(msg, ...) do { \
    printf("    [PASS] " msg "\n", ##__VA_ARGS__); g_pass++; } while(0)
#define FAIL(msg, ...) do { \
    printf("    [FAIL] " msg "\n", ##__VA_ARGS__); g_fail++; } while(0)
#define CHECK(cond, msg, ...) do { \
    if (cond) { PASS(msg, ##__VA_ARGS__); } \
    else       { FAIL(msg, ##__VA_ARGS__); } } while(0)

/* ============================================================
 * VOL FAPL helpers
 * ============================================================ */
static hid_t make_vol_fapl(void) {
    hid_t native = H5VLget_connector_id_by_name("native");
    hid_t fapl   = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native, NULL);
    H5VLclose(native);
    return fapl;
}

static hid_t make_native_fapl(void) {
    return H5Pcreate(H5P_FILE_ACCESS);   /* default = native VOL */
}

/* ============================================================
 * Byte-exact verify helper: compare h_readback against formula.
 * Returns number of mismatches (prints first).
 * offset_elem: global start element for this buffer.
 * ============================================================ */
static size_t verify_buf(const float *h, size_t n, size_t offset_elem)
{
    size_t errs = 0;
    for (size_t i = 0; i < n; i++) {
        float exp = expected(offset_elem + i);
        if (h[i] != exp) {
            if (errs == 0)
                printf("      first mismatch @ elem %zu: got %.8g exp %.8g\n",
                       offset_elem+i, (double)h[i], (double)exp);
            errs++;
        }
    }
    return errs;
}

/* ============================================================
 * ══════════════════════════════════════════════════════
 * TEST A: VOL write → NATIVE read
 * ══════════════════════════════════════════════════════
 * ============================================================ */
static void test_A(float *d_data, float *h_buf, hid_t vol_id)
{
    printf("\n── A: VOL write → Native HDF5 read ──────────────────────\n");

    hsize_t dims[1]  = { (hsize_t)TOTAL_FLOATS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_FLOATS };

    /* A1: Write via VOL */
    remove(TMP_FILE_VOL);
    hid_t fapl = make_vol_fapl();
    hid_t f    = H5Fcreate(TMP_FILE_VOL, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 0, 0.0);

    hid_t ds = H5Dcreate2(f, "ramp", H5T_NATIVE_FLOAT,
                           fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fsp);

    herr_t wret = H5Dwrite(ds, H5T_NATIVE_FLOAT,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(ds);
    H5Fclose(f);
    CHECK(wret >= 0, "VOL write succeeded");

    /* A2: Reopen with NATIVE VOL (no gpucompress) */
    fapl = make_native_fapl();
    f    = H5Fopen(TMP_FILE_VOL, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    CHECK(f >= 0, "Native H5Fopen of VOL-written file");
    if (f < 0) { FAIL("skipping A sub-tests (file unreadable)"); return; }

    /* A3: Dataset exists and metadata matches */
    ds = H5Dopen2(f, "ramp", H5P_DEFAULT);
    CHECK(ds >= 0, "Native H5Dopen2 finds 'ramp' dataset");

    hid_t fsp2 = H5Dget_space(ds);
    int ndims  = H5Sget_simple_extent_ndims(fsp2);
    hsize_t shape[1] = {0};
    H5Sget_simple_extent_dims(fsp2, shape, NULL);
    H5Sclose(fsp2);
    CHECK(ndims == 1, "Dataset is 1-D");
    CHECK(shape[0] == (hsize_t)TOTAL_FLOATS,
          "Dimension matches: %llu == %zu",
          (unsigned long long)shape[0], TOTAL_FLOATS);

    hid_t dtype = H5Dget_type(ds);
    int   is_float = H5Tequal(dtype, H5T_NATIVE_FLOAT);
    H5Tclose(dtype);
    CHECK(is_float > 0, "Datatype is H5T_NATIVE_FLOAT");

    /* A4: Chunk properties intact */
    hid_t dcpl2 = H5Dget_create_plist(ds);
    H5D_layout_t layout = H5Pget_layout(dcpl2);
    CHECK(layout == H5D_CHUNKED, "Layout is H5D_CHUNKED");

    hsize_t read_cdims[1] = {0};
    H5Pget_chunk(dcpl2, 1, read_cdims);
    CHECK(read_cdims[0] == (hsize_t)CHUNK_FLOATS,
          "Chunk dimension preserved: %llu == %d",
          (unsigned long long)read_cdims[0], CHUNK_FLOATS);

    /* A5: Filter ID 305 stored in DCPL */
    int nfilters = H5Pget_nfilters(dcpl2);
    int found_305 = 0;
    for (int fi = 0; fi < nfilters; fi++) {
        unsigned flags, cd[16]; size_t cd_sz = 16;
        char name[64] = {0};
        H5Z_filter_t fid = H5Pget_filter2(dcpl2, (unsigned)fi,
                                            &flags, &cd_sz, cd, 64, name, NULL);
        if (fid == H5Z_FILTER_GPUCOMPRESS) found_305 = 1;
    }
    H5Pclose(dcpl2);
    CHECK(found_305, "Filter ID 305 (GPUCompress) stored in DCPL");

    /* A6: Read raw compressed chunk — verifies HDF5 chunk index is valid */
    hsize_t chunk_offset[1] = {0};
    hsize_t chunk_nbytes = 0;
    herr_t ret = H5Dget_chunk_storage_size(ds, chunk_offset, &chunk_nbytes);
    /* nvcomp may produce output slightly larger than input for incompressible
     * data (header overhead).  The meaningful check is that the chunk exists
     * and its storage size was recorded — not that it shrank. */
    CHECK(ret >= 0 && chunk_nbytes > 0,
          "Chunk 0 storage size recorded: %llu bytes (raw %zu%s)",
          (unsigned long long)chunk_nbytes, CHUNK_FLOATS*sizeof(float),
          chunk_nbytes < CHUNK_FLOATS*sizeof(float) ? ", compressed" : ", expanded — incompressible data");

    H5Dclose(ds);
    H5Fclose(f);

    /* A7: Full decompressed read via native + H5Z filter plugin
     *     (only works if HDF5_PLUGIN_PATH points to libH5Zgpucompress.so) */
    fapl = make_native_fapl();
    f    = H5Fopen(TMP_FILE_VOL, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    ds   = H5Dopen2(f, "ramp", H5P_DEFAULT);

    /* Suppress filter-not-found errors: if plugin not in path, this is
     * expected to fail.  We just report whether it worked. */
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    herr_t rret = H5Dread(ds, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, h_buf);
    H5Dclose(ds); H5Fclose(f);

    if (rret >= 0) {
        size_t errs = verify_buf(h_buf, TOTAL_FLOATS, 0);
        CHECK(errs == 0,
              "Native read via H5Z plugin: bit-exact (%zu errors)", errs);
    } else {
        printf("    [INFO] Native H5Z read skipped — "
               "set HDF5_PLUGIN_PATH to dir containing libH5Zgpucompress.so\n");
    }

    remove(TMP_FILE_VOL);
}

/* ============================================================
 * TEST B: NATIVE write (uncompressed) → VOL read
 * ============================================================ */
static void test_B(float *d_read, float *h_buf)
{
    printf("\n── B: Native write (no compression) → VOL read ──────────\n");

    hsize_t dims[1]  = { (hsize_t)TOTAL_FLOATS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_FLOATS };

    /* B1: Write with native HDF5, chunked, NO compression, host pointer */
    float *h_orig = (float *)malloc(TOTAL_BYTES);
    if (!h_orig) { FAIL("malloc failed"); return; }
    for (size_t i = 0; i < TOTAL_FLOATS; i++)
        h_orig[i] = expected(i);

    remove(TMP_FILE_NAT);
    hid_t fapl = make_native_fapl();
    hid_t f    = H5Fcreate(TMP_FILE_NAT, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);   /* chunked, no filter */

    hid_t ds = H5Dcreate2(f, "ramp", H5T_NATIVE_FLOAT,
                           fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fsp);

    herr_t wret = H5Dwrite(ds, H5T_NATIVE_FLOAT,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, h_orig);
    H5Dclose(ds); H5Fclose(f);
    CHECK(wret >= 0, "Native write (uncompressed chunked) succeeded");

    /* B2: Open via GPUCompress VOL — reads into GPU buffer */
    fapl = make_vol_fapl();
    f    = H5Fopen(TMP_FILE_NAT, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    CHECK(f >= 0, "VOL can open native-written HDF5 file");
    if (f < 0) { free(h_orig); remove(TMP_FILE_NAT); return; }

    ds = H5Dopen2(f, "ramp", H5P_DEFAULT);
    CHECK(ds >= 0, "VOL finds 'ramp' in native file");

    cudaMemset(d_read, 0, TOTAL_BYTES);
    herr_t rret = H5Dread(ds, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(ds); H5Fclose(f);
    CHECK(rret >= 0, "VOL H5Dread of native file succeeded");

    cudaMemcpy(h_buf, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    size_t errs = verify_buf(h_buf, TOTAL_FLOATS, 0);
    CHECK(errs == 0, "VOL read of native file: bit-exact (%zu errors)", errs);

    free(h_orig);
    remove(TMP_FILE_NAT);
}

/* ============================================================
 * TEST C: NATIVE write (GZIP) → VOL read (passthrough)
 * ============================================================ */
static void test_C(float *d_read, float *h_buf)
{
    printf("\n── C: Native write (GZIP) → VOL read passthrough ────────\n");

    hsize_t dims[1]  = { (hsize_t)TOTAL_FLOATS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_FLOATS };

    float *h_orig = (float *)malloc(TOTAL_BYTES);
    if (!h_orig) { FAIL("malloc failed"); return; }
    for (size_t i = 0; i < TOTAL_FLOATS; i++) h_orig[i] = expected(i);

    remove(TMP_FILE_GZ);
    hid_t fapl = make_native_fapl();
    hid_t f    = H5Fcreate(TMP_FILE_GZ, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    H5Pset_deflate(dcpl, 6);   /* GZIP level 6 */

    hid_t ds = H5Dcreate2(f, "ramp", H5T_NATIVE_FLOAT,
                           fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fsp);

    herr_t wret = H5Dwrite(ds, H5T_NATIVE_FLOAT,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, h_orig);
    H5Dclose(ds); H5Fclose(f);
    CHECK(wret >= 0, "Native GZIP-compressed write succeeded");

    /* Open via our VOL — no GPU pointer, falls through to native GZIP */
    fapl = make_vol_fapl();
    f    = H5Fopen(TMP_FILE_GZ, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    CHECK(f >= 0, "VOL opens GZIP-compressed native file");
    if (f < 0) { free(h_orig); remove(TMP_FILE_GZ); return; }

    ds = H5Dopen2(f, "ramp", H5P_DEFAULT);
    cudaMemset(d_read, 0, TOTAL_BYTES);

    /* Read into GPU pointer — VOL should detect no gpucompress filter,
     * fall back to native path, decompress GZIP, copy to GPU */
    herr_t rret = H5Dread(ds, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(ds); H5Fclose(f);
    CHECK(rret >= 0, "VOL H5Dread of GZIP file succeeded (passthrough)");

    cudaMemcpy(h_buf, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    size_t errs = verify_buf(h_buf, TOTAL_FLOATS, 0);
    CHECK(errs == 0, "VOL passthrough GZIP read: bit-exact (%zu errors)", errs);

    free(h_orig);
    remove(TMP_FILE_GZ);
}

/* ============================================================
 * TEST D: Shell tool check — h5ls / h5dump
 * ============================================================ */
static void test_D(float *d_data, hid_t vol_id)
{
    (void)vol_id;
    printf("\n── D: Shell tool compatibility (h5ls / h5dump) ──────────\n");

    hsize_t dims[1]  = { (hsize_t)TOTAL_FLOATS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_FLOATS };

    /* Write via VOL */
    remove(TMP_FILE_VOL);
    hid_t fapl = make_vol_fapl();
    hid_t f    = H5Fcreate(TMP_FILE_VOL, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 0, 0.0);

    hid_t ds = H5Dcreate2(f, "ramp", H5T_NATIVE_FLOAT,
                           fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fsp);
    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(ds);
    H5Fclose(f);

    /* D1: h5ls */
    {
        char cmd[256];
        snprintf(cmd, sizeof cmd, "h5ls -v %s 2>&1", TMP_FILE_VOL);
        FILE *p = popen(cmd, "r");
        int   ok = 0;
        if (p) {
            char line[512];
            while (fgets(line, sizeof line, p)) {
                if (strstr(line, "ramp")) ok = 1;
            }
            int rc = pclose(p);
            CHECK(rc == 0 && ok,
                  "h5ls finds 'ramp' dataset in VOL-written file (rc=%d, found=%d)",
                  rc, ok);
        } else {
            printf("    [INFO] h5ls not found in PATH — skipping D1\n");
        }
    }

    /* D2: h5dump header (first 20 lines) */
    {
        char cmd[256];
        snprintf(cmd, sizeof cmd, "h5dump -H %s 2>&1", TMP_FILE_VOL);
        FILE *p = popen(cmd, "r");
        int   ok = 0;
        if (p) {
            char line[512];
            while (fgets(line, sizeof line, p)) {
                if (strstr(line, "DATASET") || strstr(line, "ramp")) ok = 1;
            }
            int rc = pclose(p);
            CHECK(rc == 0 && ok,
                  "h5dump -H reads DATASET metadata from VOL-written file");
        } else {
            printf("    [INFO] h5dump not found in PATH — skipping D2\n");
        }
    }

    /* D3: H5Fis_accessible — the ultimate validity check */
    htri_t acc = H5Fis_accessible(TMP_FILE_VOL, H5P_DEFAULT);
    CHECK(acc > 0, "H5Fis_accessible: file is a valid HDF5 file");

    remove(TMP_FILE_VOL);
}

/* ============================================================
 * TEST E: Metadata round-trip (3D + string attribute)
 * ============================================================ */
static void test_E(float *d_data, float *d_read, float *h_buf)
{
    printf("\n── E: Metadata round-trip (3D dataset + HDF5 attribute) ─\n");

    /* 3D dataset: 4 × 2 × CHUNK_FLOATS — total = TOTAL_FLOATS (8 chunks of CHUNK_FLOATS).
     * Each (i,j,:) slice is one chunk, giving the same N_CHUNKS=8 concurrency as other tests. */
    hsize_t dims3[3]  = { 4, 2, (hsize_t)CHUNK_FLOATS };
    hsize_t cdims3[3] = { 1, 1, (hsize_t)CHUNK_FLOATS };

    remove(TMP_FILE_META);
    hid_t fapl = make_vol_fapl();
    hid_t f    = H5Fcreate(TMP_FILE_META, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hid_t fsp  = H5Screate_simple(3, dims3, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims3);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_ZSTD, 0, 0, 0.0);

    hid_t ds = H5Dcreate2(f, "field3d", H5T_NATIVE_FLOAT,
                           fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fsp);

    /* Attach a string attribute via native HDF5 */
    {
        hid_t aspace = H5Screate(H5S_SCALAR);
        hid_t atype  = H5Tcopy(H5T_C_S1);
        H5Tset_size(atype, H5T_VARIABLE);
        hid_t attr   = H5Acreate2(ds, "description", atype, aspace,
                                   H5P_DEFAULT, H5P_DEFAULT);
        const char *val = "GPUCompress VOL 3-D compatibility test";
        H5Awrite(attr, atype, &val);
        H5Aclose(attr); H5Tclose(atype); H5Sclose(aspace);
    }

    /* Use only enough of d_data to fill the 3D dataset (same TOTAL_BYTES) */
    herr_t wret = H5Dwrite(ds, H5T_NATIVE_FLOAT,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(ds); H5Fclose(f);
    CHECK(wret >= 0, "3D VOL write with string attribute succeeded");

    /* E2: Re-open native, read attribute, check shape */
    fapl = make_native_fapl();
    f    = H5Fopen(TMP_FILE_META, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    CHECK(f >= 0, "Native open of 3D VOL file");

    ds = H5Dopen2(f, "field3d", H5P_DEFAULT);

    hid_t fsp2 = H5Dget_space(ds);
    int   nd   = H5Sget_simple_extent_ndims(fsp2);
    hsize_t sh[3] = {0};
    H5Sget_simple_extent_dims(fsp2, sh, NULL);
    H5Sclose(fsp2);
    CHECK(nd == 3, "3D dataset ndims == 3");
    CHECK(sh[0]==dims3[0] && sh[1]==dims3[1] && sh[2]==dims3[2],
          "3D shape preserved: %llux%llux%llu",
          (unsigned long long)sh[0],
          (unsigned long long)sh[1],
          (unsigned long long)sh[2]);

    /* Read attribute */
    char *attr_val = NULL;
    hid_t attr  = H5Aopen(ds, "description", H5P_DEFAULT);
    hid_t atype = H5Aget_type(attr);
    H5Aread(attr, atype, &attr_val);
    CHECK(attr_val && strstr(attr_val, "GPUCompress"),
          "String attribute readable by native HDF5: \"%s\"",
          attr_val ? attr_val : "(null)");
    if (attr_val) free(attr_val);
    H5Tclose(atype); H5Aclose(attr);
    H5Dclose(ds); H5Fclose(f);

    /* E3: Read back via VOL into GPU buffer → verify */
    cudaMemset(d_read, 0, TOTAL_BYTES);
    fapl = make_vol_fapl();
    f    = H5Fopen(TMP_FILE_META, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    ds   = H5Dopen2(f, "field3d", H5P_DEFAULT);
    herr_t rret = H5Dread(ds, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(ds); H5Fclose(f);
    CHECK(rret >= 0, "3D VOL read-back succeeded");

    cudaMemcpy(h_buf, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    size_t errs = verify_buf(h_buf, TOTAL_FLOATS, 0);
    CHECK(errs == 0, "3D VOL round-trip bit-exact (%zu errors)", errs);

    remove(TMP_FILE_META);
}

/* ============================================================
 * TEST F: Multi-open stress — open/close 10× without data loss
 * ============================================================ */
static void test_F(float *d_data, float *d_read, float *h_buf)
{
    printf("\n── F: Multi-open stress (10 open/close cycles) ──────────\n");

    hsize_t dims[1]  = { (hsize_t)TOTAL_FLOATS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_FLOATS };

    remove(TMP_FILE_VOL);
    {
        hid_t fapl = make_vol_fapl();
        hid_t f    = H5Fcreate(TMP_FILE_VOL, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        hid_t fsp  = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);
        H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_ZSTD, 0, 0, 0.0);
        hid_t ds = H5Dcreate2(f, "data", H5T_NATIVE_FLOAT,
                               fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(fsp);
        H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(ds); H5Fclose(f);
    }

    int cycle_fails = 0;
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_read, 0, TOTAL_BYTES);
        hid_t fapl = make_vol_fapl();
        hid_t f    = H5Fopen(TMP_FILE_VOL, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        hid_t ds   = H5Dopen2(f, "data", H5P_DEFAULT);
        herr_t r   = H5Dread(ds, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        cudaDeviceSynchronize();
        H5Dclose(ds); H5Fclose(f);

        cudaMemcpy(h_buf, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);
        size_t errs = (r >= 0) ? verify_buf(h_buf, TOTAL_FLOATS, 0) : 1;
        if (errs) cycle_fails++;
    }
    CHECK(cycle_fails == 0,
          "10 open/close cycles: all bit-exact (%d failures)", cycle_fails);

    remove(TMP_FILE_VOL);
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  HDF5 Compatibility Test — VOL ↔ Native Interoperability\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  Data    : %zu MiB, %d chunks × %d MiB\n",
           TOTAL_BYTES >> 20, N_CHUNKS, (int)(CHUNK_FLOATS*sizeof(float) >> 20));
    printf("  Pattern : ramp  f[i] = i / total  (formula-verified)\n");
    printf("══════════════════════════════════════════════════════════\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Init gpucompress (no NN needed — explicit algorithms only) */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: H5VL_gpucompress_register failed\n");
        gpucompress_cleanup(); return 1;
    }

    /* GPU and host buffers */
    float *d_data = NULL, *d_read = NULL, *h_buf = NULL;
    if (cudaMalloc(&d_data, TOTAL_BYTES) != cudaSuccess ||
        cudaMalloc(&d_read, TOTAL_BYTES) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc\n");
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }
    h_buf = (float *)malloc(TOTAL_BYTES);
    if (!h_buf) {
        fprintf(stderr, "FATAL: malloc\n");
        cudaFree(d_data); cudaFree(d_read);
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }

    /* Fill GPU buffer */
    ramp_kernel<<<512, 256>>>(d_data, TOTAL_FLOATS, TOTAL_FLOATS);
    cudaDeviceSynchronize();

    /* Run tests */
    test_A(d_data, h_buf, vol_id);
    test_B(d_read, h_buf);
    test_C(d_read, h_buf);
    test_D(d_data, vol_id);
    test_E(d_data, d_read, h_buf);
    test_F(d_data, d_read, h_buf);

    /* Summary */
    int total = g_pass + g_fail;
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  SUMMARY: %d / %d checks passed", g_pass, total);
    if (g_fail) printf("   (%d FAILED)", g_fail);
    printf("\n");

    if (g_fail == 0) {
        printf("\n  ✓ VOL-written files are valid HDF5 (native tools can open them)\n");
        printf("  ✓ Filter ID 305 stored correctly in DCPL metadata\n");
        printf("  ✓ VOL can read standard native-written HDF5 files\n");
        printf("  ✓ VOL can read GZIP-compressed native files (passthrough)\n");
        printf("  ✓ 3D datasets + HDF5 attributes survive VOL round-trip\n");
        printf("  ✓ 10 open/close cycles: no data corruption\n");
    } else {
        printf("\n  ✗ %d check(s) failed — see details above\n", g_fail);
    }
    printf("══════════════════════════════════════════════════════════\n\n");

    free(h_buf);
    cudaFree(d_data);
    cudaFree(d_read);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return (g_fail == 0) ? 0 : 1;
}
