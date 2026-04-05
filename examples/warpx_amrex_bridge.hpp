/**
 * @file warpx_amrex_bridge.hpp
 * @brief Header-only bridge between WarpX/AMReX and GPUCompress
 *
 * Writes compressed HDF5 using GPUCompress VOL connector. WarpX stores
 * electromagnetic fields (E, B), current density (J), and charge density
 * (rho) in AMReX MultiFabs on GPU. This bridge borrows those device
 * pointers and compresses them in-situ without host round-trips.
 *
 * Follows the same pattern as nyx_amrex_bridge.hpp.
 */

#ifndef WARPX_AMREX_BRIDGE_HPP
#define WARPX_AMREX_BRIDGE_HPP

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_VisMF.H>

#include <hdf5.h>
#include <cuda_runtime.h>
#include <string>
#include <cstdio>

namespace gpucompress_warpx_bridge {

/**
 * Initialize GPUCompress + HDF5 VOL connector. Call once.
 * Returns the FAPL with VOL configured, or H5I_INVALID_HID on failure.
 */
inline hid_t init(const char* weights_path)
{
    gpucompress_error_t err = gpucompress_init(weights_path);
    if (err != GPUCOMPRESS_SUCCESS) return H5I_INVALID_HID;

    H5Z_gpucompress_register();
    hid_t vol_id = H5VL_gpucompress_register();
    (void)vol_id;

    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    return fapl;
}

/**
 * Write GPU-resident data to compressed HDF5.
 * HDF5 chunking + VOL connector handles compression transparently.
 */
inline void write_gpu_to_hdf5(const char* filename, const char* dset_name,
                               const void* d_data, size_t n_elements,
                               hid_t h5type, size_t chunk_elements,
                               hid_t fapl,
                               gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
                               double error_bound = 0.0)
{
    if (chunk_elements > n_elements) chunk_elements = n_elements;
    hsize_t dims[1]  = { (hsize_t)n_elements };
    hsize_t cdims[1] = { (hsize_t)chunk_elements };

    hid_t fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (fid < 0) return;

    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int shuffle_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    unsigned int preproc = 0;
    if (error_bound > 0.0) preproc = GPUCOMPRESS_PREPROC_QUANTIZE;
    H5Pset_gpucompress(dcpl, algo, preproc, shuffle_size, error_bound);

    hid_t dset = H5Dcreate2(fid, dset_name, h5type, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset >= 0) {
        /* d_data is a CUDA device pointer -- VOL detects this and
         * compresses each chunk on GPU, writes pre-compressed bytes */
        H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(dset);
    }

    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(fid);
}

/**
 * Read back compressed HDF5 and verify bitwise against original GPU data.
 * Returns 0 on success (bitwise match), -1 on mismatch or error.
 */
inline int verify_gpu_hdf5(const char* filename, const char* dset_name,
                            const void* d_original, size_t n_elements,
                            hid_t h5type, hid_t fapl)
{
    hid_t fid = H5Fopen(filename, H5F_ACC_RDONLY, fapl);
    if (fid < 0) return -1;

    hid_t dset = H5Dopen2(fid, dset_name, H5P_DEFAULT);
    if (dset < 0) { H5Fclose(fid); return -1; }

    size_t elem_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    size_t total_bytes = n_elements * elem_size;

    void* d_readback = nullptr;
    cudaMalloc(&d_readback, total_bytes);

    H5Dread(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    cudaDeviceSynchronize();

    H5Dclose(dset);
    H5Fclose(fid);

    /* Bitwise comparison on host */
    std::vector<char> h_orig(total_bytes);
    std::vector<char> h_read(total_bytes);
    cudaMemcpy(h_orig.data(), d_original, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read.data(), d_readback, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_readback);

    int result = memcmp(h_orig.data(), h_read.data(), total_bytes);
    return (result == 0) ? 0 : -1;
}

/**
 * Write a single WarpX field MultiFab to compressed HDF5 files.
 * One HDF5 file per FArrayBox, each with chunked+compressed dataset.
 *
 * WarpX stores E, B, J as separate MultiFabs (one per component direction
 * on a staggered Yee grid). This function handles one MultiFab at a time.
 *
 * @param dir          Output directory
 * @param field_name   Field label (e.g. "Ex", "By", "jz", "rho")
 * @param mf           MultiFab (GPU-resident data)
 * @param fapl         File access property list from init()
 * @param chunk_bytes  HDF5 chunk size in bytes (default 4 MiB)
 * @param algo         Compression algorithm
 * @param error_bound  0.0 = lossless
 * @param verify       If true, read back and verify bitwise
 * @return Total original bytes written
 */
inline long write_field_compressed(
    const std::string& dir,
    const std::string& field_name,
    const amrex::MultiFab& mf,
    hid_t fapl,
    size_t chunk_bytes = 4 * 1024 * 1024,
    gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
    double error_bound = 0.0,
    bool verify = false)
{
    amrex::Gpu::streamSynchronize();

    int ncomp = mf.nComp();
    long total_original = 0;

    if (amrex::ParallelDescriptor::IOProcessor())
        amrex::UtilCreateDirectory(dir, 0755);
    amrex::ParallelDescriptor::Barrier();

    hid_t h5type = (sizeof(amrex::Real) == 8) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    size_t elem_size = sizeof(amrex::Real);
    size_t chunk_elems = chunk_bytes / elem_size;

    int fab_idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi, ++fab_idx)
    {
        const amrex::FArrayBox& fab = mf[mfi];
        long ncells = mfi.validbox().numPts();
        size_t n_elements = (size_t)ncells * ncomp;
        size_t fab_bytes = n_elements * elem_size;

        const void* d_ptr = (const void*)fab.dataPtr();

        char fname[256];
        snprintf(fname, sizeof(fname), "%s/%s_fab_%04d.h5",
                 dir.c_str(), field_name.c_str(), fab_idx);

        H5VL_gpucompress_reset_stats();
        write_gpu_to_hdf5(fname, "data", d_ptr, n_elements,
                          h5type, chunk_elems, fapl, algo, error_bound);

        cudaDeviceSynchronize();

        if (verify) {
            int vrc = verify_gpu_hdf5(fname, "data", d_ptr, n_elements,
                                       h5type, fapl);
            if (vrc != 0) {
                amrex::Print() << "[GPUCompress] VERIFY FAILED: "
                               << field_name << "_fab_" << fab_idx
                               << " bitwise mismatch!\n";
                amrex::Abort("GPUCompress lossless verification failed");
            }
        }

        total_original += fab_bytes;
    }

    return total_original;
}

/**
 * Print compression statistics.
 */
inline void print_stats(const std::string& label,
                        long original_bytes, long compressed_bytes,
                        double elapsed_ms = 0.0)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    double ratio = (compressed_bytes > 0)
                 ? (double)original_bytes / compressed_bytes : 0.0;
    double orig_mb = original_bytes / (1024.0 * 1024.0);
    double comp_mb = compressed_bytes / (1024.0 * 1024.0);

    amrex::Print() << "[GPUCompress] " << label << ": "
                   << orig_mb << " MB -> " << comp_mb << " MB"
                   << " (ratio " << ratio << "x)";
    if (elapsed_ms > 0.0)
        amrex::Print() << " in " << elapsed_ms << " ms"
                       << " (" << (orig_mb / (elapsed_ms / 1000.0)) << " MB/s)";
    amrex::Print() << "\n";
}

} /* namespace gpucompress_warpx_bridge */

#endif /* WARPX_AMREX_BRIDGE_HPP */
