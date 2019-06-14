#include "shared.h"
#include "comms.h"
#include "mesh.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MPI
#include "mpi.h"
#endif

struct Profile compute_profile;
struct Profile comms_profile;

// Write out data for visualisation in visit
void write_to_visit(const int nx, const int ny, const int x_off,
                    const int y_off, const double* data, const char* name,
                    const int step, const double time) {
  write_to_visit_3d(nx, ny, 1, x_off, y_off, 0, data, name, step, time);
}

// Write out data for visualisation in visit
void write_to_visit_3d(const int nx, const int ny, const int nz,
                       const int x_off, const int y_off, const int z_off,
                       const double* data, const char* name, const int step,
                       const double time) {
#ifdef ENABLE_VISIT_DUMPS
  char bovname[256];
  char datname[256];
  sprintf(bovname, "%s%d.bov", name, step);
  sprintf(datname, "%s%d.dat", name, step);

  FILE* bovfp = fopen(bovname, "w");
  if (!bovfp) {
    TERMINATE("Could not open file %s\n", bovname);
  }

  fprintf(bovfp, "TIME: %.8f\n", time);
  fprintf(bovfp, "DATA_FILE: %s\n", datname);
  fprintf(bovfp, "DATA_SIZE: %d %d %d\n", nx, ny, nz);
  fprintf(bovfp, "DATA_FORMAT: DOUBLE\n");
  fprintf(bovfp, "VARIABLE: density\n");
  fprintf(bovfp, "DATA_ENDIAN: LITTLE\n");
  fprintf(bovfp, "CENTERING: zone\n");

#ifdef MPI
  fprintf(bovfp, "BRICK_ORIGIN: %f %f %f.\n", (float)x_off, (float)y_off,
          (float)z_off);
#else
  fprintf(bovfp, "BRICK_ORIGIN: 0. 0. 0.\n");
#endif

  fprintf(bovfp, "BRICK_SIZE: %d %d %d\n", nx, ny, nz);
  fclose(bovfp);

  FILE* datfp = fopen(datname, "wb");
  if (!datfp) {
    TERMINATE("Could not open file %s\n", datname);
  }

  fwrite(data, sizeof(double), nx * ny * nz, datfp);
  fclose(datfp);
#endif
}

// TODO: Fix this method - shouldn't be necessary to bring the data back from
// all of the ranks, this is over the top
// This is a leaky nasty function, that really doesn't suit any of the style of
// the rest of the project, so needs immediate revisiting.
void write_all_ranks_to_visit(const int global_nx, const int global_ny,
                              const int local_nx, const int local_ny,
                              const int pad, const int x_off, const int y_off,
                              const int rank, const int nranks, int* neighbours,
                              double* local_arr, const char* name, const int tt,
                              const double elapsed_sim_time) {
#ifdef DEBUG
  if (rank == MASTER)
    printf("writing results to visit file %s\n", name);
#endif

// If MPI is enabled need to collect the data from all
#if defined(MPI)
  double* global_arr = NULL;
  double* remote_data = NULL;
  double* h_local_arr_space = NULL;
  allocate_host_data(&h_local_arr_space, local_nx * local_ny);

  double* h_local_arr = h_local_arr_space;
  copy_buffer(local_nx * local_ny, &local_arr, &h_local_arr, RECV);

  if (rank == MASTER) {
    allocate_host_data(&global_arr, global_nx * global_ny);
    remote_data = h_local_arr;
  }

  for (int ii = 0; ii < nranks; ++ii) {
    int nparams = 8;
    int dims[nparams];
    dims[0] = local_nx;
    dims[1] = local_ny;
    dims[2] = x_off;
    dims[3] = y_off;
    dims[4] = (neighbours[NORTH] == EDGE) ? 0 : pad;
    dims[5] = (neighbours[EAST] == EDGE) ? 0 : pad;
    dims[6] = (neighbours[SOUTH] == EDGE) ? 0 : pad;
    dims[7] = (neighbours[WEST] == EDGE) ? 0 : pad;

    if (rank == MASTER) {
      if (ii > MASTER) {
        MPI_Recv(&dims, nparams, MPI_INT, ii, TAG_VISIT0, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);
        allocate_host_data(&remote_data, dims[0] * dims[1]);
        MPI_Recv(remote_data, dims[0] * dims[1], MPI_DOUBLE, ii, TAG_VISIT1,
                 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }

      int lnx = dims[0];
      int lny = dims[1];
      int lx_off = dims[2];
      int ly_off = dims[3];
      int north = dims[4];
      int east = dims[5];
      int south = dims[6];
      int west = dims[7];

      // TODO: fix or remove this horrible piece
      for (int jj = south; jj < lny - north; ++jj) {
        for (int kk = west; kk < lnx - east; ++kk) {
          global_arr[(jj - south + ly_off + south) * global_nx +
                     (kk - west + lx_off + west)] = remote_data[jj * lnx + kk];
        }
      }

      if (ii > MASTER) {
        deallocate_data(remote_data);
      }
    } else if (ii == rank) {
      MPI_Send(&dims, nparams, MPI_INT, MASTER, TAG_VISIT0, MPI_COMM_WORLD);
      MPI_Send(h_local_arr, dims[0] * dims[1], MPI_DOUBLE, MASTER, TAG_VISIT1,
               MPI_COMM_WORLD);
    }
    barrier();
  }
  if (rank == MASTER) {
    write_to_visit(global_nx, global_ny, 0, 0, global_arr, name, tt,
                   elapsed_sim_time);
  }
  barrier();

  deallocate_data(h_local_arr_space);
#else
  write_to_visit(global_nx, global_ny, 0, 0, local_arr, name, tt,
                 elapsed_sim_time);
#endif
}
