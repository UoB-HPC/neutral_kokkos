#ifndef __COMMSHDR
#define __COMMSHDR

#pragma once

#include "mesh.h"
#include "shared.h"
//#include "umesh.h"
//#include <complex.h>
//#undef complex

#define MASTER 0        // The master rank for MPI
#define NVARS_TO_COMM 4 // This is just the max of HOT and WET

enum { NO_PACK, PACK }; // Whether a buffer should be packed and communicated
enum {
  EDGE = -1,
  NORTH,
  EAST,
  SOUTH,
  WEST,
  FRONT,
  BACK
}; // Directional enumeration
enum {
  NO_INVERT,
  INVERT_X,
  INVERT_Y,
  INVERT_Z
}; // Whether an inversion is required

#ifdef __cplusplus
extern "C" {
#endif

void initialise_mpi(int argc, char** argv, int* rank, int* nranks);

// Initialise the communications, potentially invoking MPI
void initialise_comms(Mesh* mesh);

// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
void decompose_2d_cartesian(const int rank, const int nranks,
                            const int global_nx, const int global_ny,
                            int* neighbours, int* local_nx, int* local_ny,
                            int* ranks_x, int* ranks_y, int* x_off, int* y_off);

// Decomposes the ranks minimising ratio of perimeter to area
void decompose_3d_cartesian(const int rank, const int nranks,
                            const int global_nx, const int global_ny,
                            const int global_nz, int* neighbours, int* local_nx,
                            int* local_ny, int* local_nz, int* ranks_x,
                            int* ranks_y, int* ranks_z, int* x_off, int* y_off,
                            int* z_off);

// Reduces the value across all ranks and returns the sum
double reduce_all_sum(double local_val);

// Reduces the value across all ranks and returns minimum result
double reduce_all_min(double local_val);

// Reduces value from all ranks to master
double reduce_to_master(double local_val);

// Performs an all to all communication
void all_to_all(const int len, double* a, double* b);

// Performs an all to all communication of complex data
void all_to_all_complex(const int len, double _Complex* a, double _Complex* b);

// Performs a complex scatter
void scatter_complex(const int len, _Complex double* send,
                     _Complex double* recv);

// Performs a complex gather
void gather_complex(const int len, _Complex double* send,
                    _Complex double* recv);

// Performs a complex gather from all ranks
void all_gather_complex(const int len, _Complex double* send,
                        _Complex double* recv);

// Performs a non-blocking mpi send
void non_block_send(double* buffer_out, const int len, const int to,
                    const int tag, const int req_index);

// Performs a non-blocking mpi recv
void non_block_recv(double* buffer_in, const int len, const int from,
                    const int tag, const int req_index);

// Waits on any queued messages
void wait_on_messages(const int nmessages);

// Performs an MPI barrier
void barrier();

// Enforce reflective boundary conditions on the problem state
void handle_boundary_2d(const int nx, const int ny, Mesh* mesh, Kokkos::View<double *> arr,
                        const int invert, const int pack);

// Finalise the communications
void finalise_comms();

#ifdef __cplusplus
}
#endif

#endif
