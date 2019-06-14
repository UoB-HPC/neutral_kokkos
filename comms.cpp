#include "comms.h"
#include "shared.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MPI
#include "mpi.h"
#endif

struct mpi_message_state {
#ifdef MPI
  MPI_Request req[2 * NNEIGHBOURS];
#endif
} msg_state;

void initialise_mpi(int argc, char** argv, int* rank, int* nranks) {
#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, nranks);
#endif
}

// Initialise the communications, potentially invoking MPI
void initialise_comms(Mesh* mesh) {
  for (int ii = 0; ii < NNEIGHBOURS; ++ii) {
    mesh->neighbours[ii] = EDGE;
  }

#ifdef APP_3D
  decompose_3d_cartesian(mesh->rank, mesh->nranks, mesh->global_nx,
                         mesh->global_ny, mesh->global_nz, mesh->neighbours,
                         &mesh->local_nx, &mesh->local_ny, &mesh->local_nz,
                         &mesh->ranks_x, &mesh->ranks_y, &mesh->ranks_z,
                         &mesh->x_off, &mesh->y_off, &mesh->z_off);
#else
  decompose_2d_cartesian(mesh->rank, mesh->nranks, mesh->global_nx,
                         mesh->global_ny, mesh->neighbours, &mesh->local_nx,
                         &mesh->local_ny, &mesh->ranks_x, &mesh->ranks_y,
                         &mesh->x_off, &mesh->y_off);
#endif

  // Add on the halo padding to the local mesh
  mesh->local_nx += 2 * mesh->pad;
  mesh->local_ny += 2 * mesh->pad;
  mesh->local_nz += 2 * mesh->pad;

  if (mesh->rank == MASTER) {
#ifdef APP_3D
    printf("Problem dimensions %dx%dx%d for %d iterations.\n", mesh->global_nx,
           mesh->global_ny, mesh->global_nz, mesh->niters);
#else
    printf("Problem dimensions %dx%d for %d iterations.\n", mesh->global_nx,
           mesh->global_ny, mesh->niters);
#endif
  }
}

#ifdef MPI
static inline double all_reduce(double local_val, MPI_Op op) {
  double global_val = local_val;
  START_PROFILING(&compute_profile);
  MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
  STOP_PROFILING(&compute_profile, "communications");
  return global_val;
}
#endif

// Reduces the value across all ranks and returns minimum result
double reduce_all_min(double local_val) {
  double global_val = local_val;
#ifdef MPI
  global_val = all_reduce(local_val, MPI_MIN);
#endif
  return global_val;
}

// Reduces the value across all ranks and returns the sum
double reduce_all_sum(double local_val) {
  double global_val = local_val;
#ifdef MPI
  global_val = all_reduce(local_val, MPI_SUM);
#endif
  return global_val;
}

// Reduce across ranks into master
double reduce_to_master(double local_val) {
  double global_val = local_val;

#ifdef MPI
  MPI_Reduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MASTER,
             MPI_COMM_WORLD);
#endif

  return global_val;
}

// Performs an all to all communication
void all_to_all(const int len, double* a, double* b) {
#ifdef MPI
  MPI_Alltoall(b, len, MPI_DOUBLE, a, len, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
}

// Performs an all to all communication of complex data
void all_to_all_complex(const int len, double _Complex* a, double _Complex* b) {
#ifdef MPI
  MPI_Alltoall(b, len, MPI_C_DOUBLE_COMPLEX, a, len, MPI_C_DOUBLE_COMPLEX,
               MPI_COMM_WORLD);
#endif
}

// Performs a complex scatter
void scatter_complex(const int len, _Complex double* send,
                     _Complex double* recv) {
#ifdef MPI
  MPI_Scatter(send, len, MPI_C_DOUBLE_COMPLEX, recv, len, MPI_C_DOUBLE_COMPLEX,
              MASTER, MPI_COMM_WORLD);
#else
  for (int i = 0; i < len; ++i) {
    recv[i] = send[i];
  }
#endif
}

// Performs a complex gather
void gather_complex(const int len, _Complex double* send,
                    _Complex double* recv) {
#ifdef MPI
  MPI_Gather(send, len, MPI_C_DOUBLE_COMPLEX, recv, len, MPI_C_DOUBLE_COMPLEX,
             MASTER, MPI_COMM_WORLD);
#else
  for (int i = 0; i < len; ++i) {
    recv[i] = send[i];
  }
#endif
}

// Performs a complex gather from all ranks
void all_gather_complex(const int len, _Complex double* send,
                        _Complex double* recv) {
#ifdef MPI
  MPI_Allgather(send, len, MPI_C_DOUBLE_COMPLEX, recv, len,
                MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
#else
  for (int i = 0; i < len; ++i) {
    recv[i] = send[i];
  }
#endif
}

// Performs an mpi barrier
void barrier() {
#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

// Performs a non-blocking mpi send
void non_block_send(double* buffer_out, const int len, const int to,
                    const int tag, const int req_index) {
#ifdef MPI
  MPI_Isend(buffer_out, len, MPI_DOUBLE, to, tag, MPI_COMM_WORLD,
            &msg_state.req[req_index]);
#endif
}

// Performs a non-blocking mpi recv
void non_block_recv(double* buffer_in, const int len, const int from,
                    const int tag, const int req_index) {
#ifdef MPI
  MPI_Irecv(buffer_in, len, MPI_DOUBLE, from, tag, MPI_COMM_WORLD,
            &msg_state.req[req_index]);
#endif
}

// Waits on any queued messages
void wait_on_messages(const int nmessages) {
#ifdef MPI
  MPI_Waitall(nmessages, msg_state.req, MPI_STATUSES_IGNORE);
#endif
}

// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
void decompose_2d_cartesian(const int rank, const int nranks,
                            const int global_nx, const int global_ny,
                            int* neighbours, int* local_nx, int* local_ny,
                            int* ranks_x, int* ranks_y, int* x_off,
                            int* y_off) {
#if defined(TILES)
  int found_even = 0;
  float mratio = 0.0f;
#ifdef DEBUG
  if (rank == MASTER) {
    printf("using tiles decomposition\n");
  }
#endif

  const int sqr_nranks = sqrtf(nranks);

  // Determine decomposition that minimises perimeter to area ratio
  for (int ff = 1; ff <= sqr_nranks; ++ff) {
    if (nranks % ff) {
      continue;
    }

    // If load balance is preferred then prioritise even split over ratio
    // Test if this split evenly decomposes into the mesh
    const int even_split_ff_x =
        (global_nx % ff == 0 && global_ny % (nranks / ff) == 0);
    const int even_split_ff_y =
        (global_nx % (nranks / ff) == 0 && global_ny % ff == 0);
    const int new_ranks_x = even_split_ff_x ? ff : nranks / ff;
    const int new_ranks_y = even_split_ff_x ? nranks / ff : ff;
    const int is_even = even_split_ff_x || even_split_ff_y;
    found_even |= (LOAD_BALANCE && is_even);

    const float potential_ratio =
        (2 * (new_ranks_x + new_ranks_y)) / (float)(new_ranks_x * new_ranks_y);

    // Update if we minimise the ratio further, only if we don't care about load
    // balancing or have found an even split
    if ((found_even <= is_even) &&
        (mratio == 0.0f || potential_ratio < mratio)) {
      mratio = potential_ratio;
      // If we didn't find even split, prefer longer mesh edge on x dimension
      *ranks_x = (!found_even && new_ranks_x > new_ranks_y) ? new_ranks_y
                                                            : new_ranks_x;
      *ranks_y = (!found_even && new_ranks_x > new_ranks_y) ? new_ranks_x
                                                            : new_ranks_y;
    }
  }
#elif defined(COLS)
#ifdef DEBUG
  if (rank == MASTER) {
    printf("using col decomposition\n");
  }
#endif
  *ranks_x = nranks;
  *ranks_y = 1;
#else
#ifdef DEBUG
  if (rank == MASTER) {
    printf("using row decomposition\n");
  }
#endif
  *ranks_x = 1;
  *ranks_y = nranks;
#endif

  // Calculate the offsets up until our rank, and then fetch rank dimensions
  int off = 0;
  const int x_rank = (rank % (*ranks_x));
  for (int xx = 0; xx <= x_rank; ++xx) {
    *x_off = off;
    const int x_floor = global_nx / (*ranks_x);
    const int x_pad_req = (global_nx != (off + ((*ranks_x) - xx) * x_floor));
    *local_nx = x_pad_req ? x_floor + 1 : x_floor;
    off += *local_nx;
  }
  off = 0;
  const int y_rank = (rank / (*ranks_x));
  for (int yy = 0; yy <= y_rank; ++yy) {
    *y_off = off;
    const int y_floor = global_ny / (*ranks_y);
    const int y_pad_req = (global_ny != (off + ((*ranks_y) - yy) * y_floor));
    *local_ny = y_pad_req ? y_floor + 1 : y_floor;
    off += *local_ny;
  }

  // Calculate the surrounding ranks
  neighbours[NORTH] = (y_rank < (*ranks_y) - 1) ? rank + (*ranks_x) : EDGE;
  neighbours[EAST] = (x_rank < (*ranks_x) - 1) ? rank + 1 : EDGE;
  neighbours[SOUTH] = (y_rank > 0) ? rank - (*ranks_x) : EDGE;
  neighbours[WEST] = (x_rank > 0) ? rank - 1 : EDGE;

#ifdef DEBUG
  printf("Rank: %d, Dimensions: %d %d, Neighbours: %d %d %d %d\n", rank,
         *local_nx, *local_ny, neighbours[NORTH], neighbours[EAST],
         neighbours[SOUTH], neighbours[WEST]);
#endif
}

// Decomposes the ranks minimising ratio of perimeter to area
void decompose_3d_cartesian(const int rank, const int nranks,
                            const int global_nx, const int global_ny,
                            const int global_nz, int* neighbours, int* local_nx,
                            int* local_ny, int* local_nz, int* ranks_x,
                            int* ranks_y, int* ranks_z, int* x_off, int* y_off,
                            int* z_off) {
  float min_sa_to_vol = 0.0f;

  // Determine decomposition that minimises surface area to volume ratio
  for (int split_z = 1; split_z <= cbrt((double)nranks); ++split_z) {
    for (int split_y = 1; split_y <= cbrt((double)nranks); ++split_y) {
      for (int split_x = 1; split_x <= nranks; ++split_x) {
        // Factorise the number of ranks
        if (nranks % split_x || (nranks / split_x) % split_y ||
            (nranks % (split_x * split_y)))
          continue;

        // Calculate the surface are to volume ratio of the rank split
        const float sa_to_vol = (2.0 * (split_x + split_y + split_z)) /
                                (float)(split_x * split_y * split_z);

        // TODO: MINIMISE THE RATIO OF EACH EDGE PAIR ON THE RANKS AND MESH
        // TO BETTER DECOMPOSE IRREGULAR PROBLEM SHAPES
        if (min_sa_to_vol == 0.0f || sa_to_vol < min_sa_to_vol) {
          min_sa_to_vol = sa_to_vol;
          // Choose edges so that x > y > z for preferred data access
          if (split_x >= split_y && split_x >= split_z) {
            *ranks_x = split_x;
            *ranks_y = (split_y > split_z) ? split_y : split_z;
            *ranks_z = (split_y > split_z) ? split_z : split_y;
          } else if (split_y >= split_x && split_y >= split_z) {
            *ranks_x = split_y;
            *ranks_y = (split_x > split_z) ? split_x : split_z;
            *ranks_z = (split_x > split_z) ? split_z : split_x;
          } else if (split_z >= split_x && split_z >= split_y) {
            *ranks_x = split_z;
            *ranks_y = (split_x > split_y) ? split_x : split_y;
            *ranks_z = (split_x > split_y) ? split_y : split_x;
          }
        }
      }
    }
  }

  // TODO: Seems refactorable
  // Calculate the offsets up until our rank, and then fetch rank dimensions
  int off = 0;
  const int x_rank = (rank % (*ranks_x));
  for (int xx = 0; xx <= x_rank; ++xx) {
    *x_off = off;
    const int x_floor = global_nx / (*ranks_x);
    const int x_pad_req = (global_nx != (off + ((*ranks_x) - xx) * x_floor));
    *local_nx = x_pad_req ? x_floor + 1 : x_floor;
    off += *local_nx;
  }
  off = 0;
  const int y_rank = ((rank / (*ranks_x)) % (*ranks_y));
  for (int yy = 0; yy <= y_rank; ++yy) {
    *y_off = off;
    const int y_floor = global_ny / (*ranks_y);
    const int y_pad_req = (global_ny != (off + ((*ranks_y) - yy) * y_floor));
    *local_ny = y_pad_req ? y_floor + 1 : y_floor;
    off += *local_ny;
  }
  off = 0;
  const int z_rank = (rank / ((*ranks_x) * (*ranks_y)));
  for (int zz = 0; zz <= z_rank; ++zz) {
    *z_off = off;
    const int z_floor = global_nz / (*ranks_z);
    const int z_pad_req = (global_nz != (off + ((*ranks_z) - zz) * z_floor));
    *local_nz = z_pad_req ? z_floor + 1 : z_floor;
    off += *local_nz;
  }

  // Calculate the surrounding ranks
  neighbours[SOUTH] = (y_rank > 0) ? rank - (*ranks_x) : EDGE;
  neighbours[WEST] = (x_rank > 0) ? rank - 1 : EDGE;
  neighbours[FRONT] = (z_rank > 0) ? rank - ((*ranks_x) * (*ranks_y)) : EDGE;
  neighbours[NORTH] = (y_rank < (*ranks_y) - 1) ? rank + (*ranks_x) : EDGE;
  neighbours[EAST] = (x_rank < (*ranks_x) - 1) ? rank + 1 : EDGE;
  neighbours[BACK] =
      (z_rank < (*ranks_z) - 1) ? rank + ((*ranks_x) * (*ranks_y)) : EDGE;

#ifdef DEBUG
  printf("Rank: %d, Dimensions: %d %d %d, Neighbours: %d %d %d %d %d %d\n",
         rank, *local_nx, *local_ny, *local_nz, neighbours[NORTH],
         neighbours[EAST], neighbours[BACK], neighbours[SOUTH],
         neighbours[WEST], neighbours[FRONT]);
#endif
}

// Decompose the unstructured space
#if 0
void decompose_unstructured_mesh(
    const int rank, const int nranks, const int ncells, const int nnodes, 
    double* cell_centroids_x, double* cell_centroids_y, int* node_neighbours) 
{
  const int carry = ncells%nranks;

  // Account for uneven decomposition 
  int ncells_per_rank = ncells/nranks;
  if(rank < carry) {
    ncells_per_rank++;
  }

  // As with structured algorithm, perform the partitioning for every rank
  for(int rr = 0; rr < nranks; ++rr) {
    // This is essentially our chance to make sure that the memory accesses
    // are correct irrespective of the ordering of the mesh in mesh file
    for(int cc = 0; cc < ncells; ++cc) {
      const int nodes_off = cells_to_nodes_off[(cc)];
      const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
      const double inv_Np = 1.0/(double)nnodes_around_cell;

      double cx = 0.0;
      double cy = 0.0;
      for(int nn = 0; nn < nnodes_around_cell; ++nn) {
        const int node_index = cells_to_nodes[(nodes_off)+(nn)];
        cx += nodes_x0[(node_index)]*inv_Np;
        cy += nodes_y0[(node_index)]*inv_Np;
      }
      cell_centroids_x[(cc)] = cx;
      cell_centroids_y[(cc)] = cy;
    }

    // Loop over all of the cell centroids to determine the closest
    for(int cc = 0; cc < ncells; ++cc) {

    }

    // Only complete up until our rank
    if(rank == rr) {
      break;
    }
  }
}
#endif // if 0

// Finalise the communications
void finalise_comms() {
#ifdef MPI
  MPI_Finalize();
#endif
}
