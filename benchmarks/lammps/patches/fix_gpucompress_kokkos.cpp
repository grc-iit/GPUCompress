/* ----------------------------------------------------------------------
   LAMMPS fix for GPU-accelerated compression via GPUCompress.

   Borrows KOKKOS device pointers and writes compressed HDF5 using
   GPUCompress's HDF5 VOL connector. Zero simulation source changes
   beyond this fix file.
------------------------------------------------------------------------- */

#include "fix_gpucompress_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos_type.h"
#include "update.h"
#include "comm.h"
#include "error.h"
#include "memory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

/* GPUCompress bridge — linked via cmake */
#include "lammps_gpucompress_udf.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixGPUCompressKokkos::FixGPUCompressKokkos(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR, "Illegal fix gpucompress command");

  /* Parse: fix ID group gpucompress N [positions] [velocities] [forces] */
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "fix gpucompress: N must be > 0");

  dump_every = nevery;
  dump_positions = 0;
  dump_velocities = 0;
  dump_forces = 0;

  for (int i = 4; i < narg; i++) {
    if (strcmp(arg[i], "positions") == 0) dump_positions = 1;
    else if (strcmp(arg[i], "velocities") == 0) dump_velocities = 1;
    else if (strcmp(arg[i], "forces") == 0) dump_forces = 1;
    else error->all(FLERR, "fix gpucompress: unknown field");
  }

  /* Default: dump all if nothing specified */
  if (!dump_positions && !dump_velocities && !dump_forces) {
    dump_positions = 1;
    dump_velocities = 1;
    dump_forces = 1;
  }

  algo_name = getenv("GPUCOMPRESS_ALGO");
  if (!algo_name) algo_name = "auto";

  const char *venv = getenv("GPUCOMPRESS_VERIFY");
  verify = (venv && atoi(venv)) ? 1 : 0;

  gpuc_ready = 0;
}

FixGPUCompressKokkos::~FixGPUCompressKokkos()
{
  /* Don't finalize here — CUDA context may already be torn down.
   * GPUCompress cleanup happens via atexit or OS process teardown. */
}

/* ---------------------------------------------------------------------- */

int FixGPUCompressKokkos::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGPUCompressKokkos::init()
{
  /* Initialize GPUCompress on first call */
  if (!gpuc_ready) {
    const char *weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "model.nnwt";  /* set GPUCOMPRESS_WEIGHTS env var */

    const char *policy = getenv("GPUCOMPRESS_POLICY");
    if (!policy) policy = "ratio";

    int rc = gpucompress_lammps_init(weights, policy);
    if (rc == 0) {
      gpuc_ready = 1;
      if (comm->me == 0) {
        fprintf(stdout, "[GPUCompress-LAMMPS] Initialized: algo=%s policy=%s verify=%d\n",
                algo_name, policy, verify);
        fprintf(stdout, "[GPUCompress-LAMMPS] Fields: pos=%d vel=%d force=%d every=%d\n",
                dump_positions, dump_velocities, dump_forces, dump_every);
        fflush(stdout);
      }
    } else {
      if (comm->me == 0)
        fprintf(stderr, "[GPUCompress-LAMMPS] Init FAILED\n");
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixGPUCompressKokkos::setup(int /* vflag */)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixGPUCompressKokkos::end_of_step()
{
  if (!gpuc_ready) return;

  bigint ntimestep = update->ntimestep;

  /* Only dump at specified intervals */
  if (ntimestep % dump_every != 0) return;

  auto *atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (!atomKK) {
    if (comm->me == 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] Not running with KOKKOS!\n");
    return;
  }

  int nlocal = atom->nlocal;
  int elem_bytes = sizeof(KK_FLOAT);  /* auto-detect: 4 for SINGLE, 8 for DOUBLE */

  /* Create output directory */
  char dir[256];
  snprintf(dir, sizeof(dir), "gpuc_step_%010lld", (long long)ntimestep);
  if (comm->me == 0) mkdir(dir, 0755);
  MPI_Barrier(world);

  char fname[512];
  int rank = comm->me;
  int rc;

  /* Sync device views to ensure data is current */
  atomKK->sync(LAMMPS_NS::Device, X_MASK | V_MASK | F_MASK);

  if (dump_positions) {
    /* Get raw CUDA device pointer from KOKKOS view
     * k_x is a DualView; view_device() returns the device Kokkos::View
     * .data() returns the raw pointer to the underlying CUDA allocation */
    auto d_x = atomKK->k_x.view_device();
    const void *d_ptr = (const void *)d_x.data();
    size_t n_elements = (size_t)nlocal * 3;

    snprintf(fname, sizeof(fname), "%s/x_rank%04d.h5", dir, rank);
    rc = gpucompress_lammps_write_field(fname, "positions", d_ptr,
                                         n_elements, elem_bytes,
                                         algo_name, 0.0, verify);
    if (comm->me == 0 && rc != 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] positions write failed\n");
  }

  if (dump_velocities) {
    auto d_v = atomKK->k_v.view_device();
    const void *d_ptr = (const void *)d_v.data();
    size_t n_elements = (size_t)nlocal * 3;

    snprintf(fname, sizeof(fname), "%s/v_rank%04d.h5", dir, rank);
    rc = gpucompress_lammps_write_field(fname, "velocities", d_ptr,
                                         n_elements, elem_bytes,
                                         algo_name, 0.0, verify);
    if (comm->me == 0 && rc != 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] velocities write failed\n");
  }

  if (dump_forces) {
    auto d_f = atomKK->k_f.view_device();
    const void *d_ptr = (const void *)d_f.data();
    size_t n_elements = (size_t)nlocal * 3;

    snprintf(fname, sizeof(fname), "%s/f_rank%04d.h5", dir, rank);
    rc = gpucompress_lammps_write_field(fname, "forces", d_ptr,
                                         n_elements, elem_bytes,
                                         algo_name, 0.0, verify);
    if (comm->me == 0 && rc != 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] forces write failed\n");
  }

  if (comm->me == 0) {
    int nfields = dump_positions + dump_velocities + dump_forces;
    double mb = (double)nlocal * 3 * elem_bytes * nfields / (1024.0 * 1024.0);
    fprintf(stdout, "[GPUCompress-LAMMPS] Step %lld: wrote %d fields (%.1f MB/rank) algo=%s\n",
            (long long)ntimestep, nfields, mb, algo_name);
    fflush(stdout);
  }
}
