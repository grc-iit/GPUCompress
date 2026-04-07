/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS fix for GPU-accelerated compression via GPUCompress.

   Accesses KOKKOS device arrays (positions, velocities, forces)
   and writes compressed HDF5 via GPUCompress's HDF5 VOL connector.
   Zero-copy: device pointers from Kokkos::View::data() are passed
   directly to the VOL connector.

   Usage in LAMMPS input script:
     fix gpuc all gpucompress 100 positions velocities forces
   where 100 = dump every 100 steps

   Environment variables:
     GPUCOMPRESS_ALGO     - algorithm (lz4/snappy/zstd/auto/etc.)
     GPUCOMPRESS_POLICY   - NN ranking policy (speed/balanced/ratio)
     GPUCOMPRESS_VERIFY   - 1 to enable lossless verification
     GPUCOMPRESS_WEIGHTS  - path to NN model weights
     GPUCOMPRESS_SGD      - 0/1: enable online SGD (default 1 for auto)
     GPUCOMPRESS_EXPLORE  - 0/1: enable exploration (default 1 for auto)
     GPUCOMPRESS_ERROR_BOUND - relative error bound for lossy quantization
                               (default 0.0 = lossless). Must be > 0 to
                               populate per-chunk PSNR data.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(gpucompress,FixGPUCompressKokkos);
// clang-format on
#else

#ifndef LMP_FIX_GPUCOMPRESS_KOKKOS_H
#define LMP_FIX_GPUCOMPRESS_KOKKOS_H

#include "fix.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class FixGPUCompressKokkos : public Fix {
 public:
  FixGPUCompressKokkos(class LAMMPS *, int, char **);
  ~FixGPUCompressKokkos() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;

 private:
  int dump_every;          /* steps between compressed dumps */
  int dump_positions;      /* 1 if writing positions */
  int dump_velocities;     /* 1 if writing velocities */
  int dump_forces;         /* 1 if writing forces */

  const char *algo_name;
  int verify;
  int dump_raw_fields;
  int log_chunks;         /* 1 if writing per-chunk CSV diagnostics */
  FILE *tc_csv;           /* timestep_chunks CSV handle */
  FILE *ranking_csv;      /* ranking CSV (top1_regret) */
  FILE *ranking_costs_csv;
  int write_count;        /* running write counter for CSV timestep col */
  int total_writes;       /* estimated total writes for milestone check */
  float cw0, cw1, cw2;   /* cost model weights for ranking profiler */
  int do_sgd;            /* 1 if online SGD is enabled (GPUCOMPRESS_SGD) */
  int do_explore;        /* 1 if exploration is enabled (GPUCOMPRESS_EXPLORE) */
  double error_bound;    /* relative error bound for lossy quantization;
                            0.0 = lossless. Set from GPUCOMPRESS_ERROR_BOUND
                            env var. Must be > 0 for the NN to pick quantized
                            configs and populate per-chunk PSNR data. */
  int gpuc_ready;
};

}    // namespace LAMMPS_NS

#endif
#endif
