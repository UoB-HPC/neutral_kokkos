#include "comms.h"
#include "mesh.h"
#include "shared.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary_2d(const int nx, const int ny, Mesh* mesh, Kokkos::View<double *> arr,
                        const int invert, const int pack) {
  START_PROFILING(&comms_profile);

    // Kokkos::parallel_for(Kokkos::RangePolicy< ExecSpace >( 0, N ), KOKKOS_LAMBDA (int i) {
    //     local_buf[i] = 0;
    // });
    // Kokkos::fence();


  const int pad = mesh->pad;
  int* neighbours = mesh->neighbours;

// #ifdef MPI

//   int nmessages = 0;
//   if (pack) {
//     // Pack east and west
//     if (neighbours[EAST] != EDGE) {
//         Kokkos::parallel_for(Kokkos::RangePolicy< ExecSpace >( pad, ny-pad ), KOKKOS_LAMBDA (int ii) {
//         for (int dd = 0; dd < pad; ++dd) {
//           mesh->east_buffer_out[(ii - pad) * pad + dd] =
//               arr[(ii * nx) + (nx - 2 * pad + dd)];
//         }
//       });
//       }

//       copy_buffer(pad * ny, &mesh->east_buffer_out, &mesh->h_east_buffer_out, RECV);
//       non_block_send(h_east_buffer_out, (ny - 2 * pad) * pad, neighbours[EAST], 2,
//                      nmessages++);
//       non_block_recv(h_east_buffer_in, (ny - 2 * pad) * pad, neighbours[EAST], 3,
//                      nmessages++);
//     }

//     if (neighbours[WEST] != EDGE) {
// #pragma omp target teams distribute parallel for collapse(2)
//       for (int ii = pad; ii < ny - pad; ++ii) {
//         for (int dd = 0; dd < pad; ++dd) {
//           west_buffer_out[(ii - pad) * pad + dd] = arr[(ii * nx) + (pad + dd)];
//         }
//       }

//       copy_buffer(pad * ny, &west_buffer_out, &west_buffer_out, RECV);
//       non_block_send(west_buffer_out, (ny - 2 * pad) * pad, neighbours[WEST], 3,
//                      nmessages++);
//       non_block_recv(west_buffer_in, (ny - 2 * pad) * pad, neighbours[WEST], 2,
//                      nmessages++);
//     }

//     // Pack north and south
//     if (neighbours[NORTH] != EDGE) {
// #pragma omp target teams distribute parallel for collapse(2)
//       for (int dd = 0; dd < pad; ++dd) {
//         for (int jj = pad; jj < nx - pad; ++jj) {
//           north_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
//               arr[(ny - 2 * pad + dd) * nx + jj];
//         }
//       }

//       copy_buffer(nx * pad, &north_buffer_out, &north_buffer_out, RECV);
//       non_block_send(north_buffer_out, (nx - 2 * pad) * pad, neighbours[NORTH],
//                      1, nmessages++);
//       non_block_recv(north_buffer_in, (nx - 2 * pad) * pad, neighbours[NORTH],
//                      0, nmessages++);
//     }

//     if (neighbours[SOUTH] != EDGE) {
// #pragma omp target teams distribute parallel for collapse(2)
//       for (int dd = 0; dd < pad; ++dd) {
//         for (int jj = pad; jj < nx - pad; ++jj) {
//           south_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
//               arr[(pad + dd) * nx + jj];
//         }
//       }

//       copy_buffer(nx * pad, &south_buffer_out, &south_buffer_out, RECV);
//       non_block_send(south_buffer_out, (nx - 2 * pad) * pad, neighbours[SOUTH],
//                      0, nmessages++);
//       non_block_recv(south_buffer_in, (nx - 2 * pad) * pad, neighbours[SOUTH],
//                      1, nmessages++);
//     }

//     wait_on_messages(nmessages);

//     // Unpack east and west
//     if (neighbours[WEST] != EDGE) {
//       copy_buffer(pad * ny, &west_buffer_in, &west_buffer_in, SEND);

// #pragma omp target teams distribute parallel for collapse(2)
//       for (int ii = pad; ii < ny - pad; ++ii) {
//         for (int dd = 0; dd < pad; ++dd) {
//           arr[ii * nx + dd] = west_buffer_in[(ii - pad) * pad + dd];
//         }
//       }
//     }

//     if (neighbours[EAST] != EDGE) {
//       copy_buffer(pad * ny, &east_buffer_in, &east_buffer_in, SEND);

// #pragma omp target teams distribute parallel for collapse(2)
//       for (int ii = pad; ii < ny - pad; ++ii) {
//         for (int dd = 0; dd < pad; ++dd) {
//           arr[ii * nx + (nx - pad + dd)] =
//               east_buffer_in[(ii - pad) * pad + dd];
//         }
//       }
//     }

//     // Unpack north and south
//     if (neighbours[NORTH] != EDGE) {
//       copy_buffer(nx * pad, &north_buffer_in, &north_buffer_in, SEND);

// #pragma omp target teams distribute parallel for collapse(2)
//       for (int dd = 0; dd < pad; ++dd) {
//         for (int jj = pad; jj < nx - pad; ++jj) {
//           arr[(ny - pad + dd) * nx + jj] =
//               north_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
//         }
//       }
//     }

//     if (neighbours[SOUTH] != EDGE) {
//       copy_buffer(nx * pad, &south_buffer_in, &south_buffer_in, SEND);

// #pragma omp target teams distribute parallel for collapse(2)
//       for (int dd = 0; dd < pad; ++dd) {
//         for (int jj = pad; jj < nx - pad; ++jj) {
//           arr[dd * nx + jj] = south_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
//         }
//       }
//     }
//   }
// #endif

//         Kokkos::parallel_for(Kokkos::RangePolicy< ExecSpace >( pad, ny-pad ), KOKKOS_LAMBDA (int ii) {
//         for (int dd = 0; dd < pad; ++dd) {
//           mesh->east_buffer_out[(ii - pad) * pad + dd] =
//               arr[(ii * nx) + (nx - 2 * pad + dd)];
//         }
//       });
//       }


  // Perform the boundary reflections, potentially with the data updated from
  // neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

   // Reflect at the north
  if (neighbours[NORTH] == EDGE) {
    for (int dd = 0; dd < pad; ++dd) {
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, nx-pad ), KOKKOS_LAMBDA (int jj) {
        arr[(ny - pad + dd) * nx + jj] =
            y_inversion_coeff * arr[(ny - 1 - pad - dd) * nx + jj];
      });
    }
  }
  // reflect at the south
  if (neighbours[SOUTH] == EDGE) {
    for (int dd = 0; dd < pad; ++dd) {
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, nx-pad ), KOKKOS_LAMBDA (int jj) {
        arr[(pad - 1 - dd) * nx + jj] =
            y_inversion_coeff * arr[(pad + dd) * nx + jj];
      });
    }
  }
  // reflect at the east
  if (neighbours[EAST] == EDGE) {
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, ny-pad ), KOKKOS_LAMBDA (int ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (nx - pad + dd)] =
            x_inversion_coeff * arr[ii * nx + (nx - 1 - pad - dd)];
      }
    });
  }
  if (neighbours[WEST] == EDGE) {
// reflect at the west
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, ny-pad ), KOKKOS_LAMBDA (int ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (pad - 1 - dd)] =
            x_inversion_coeff * arr[ii * nx + (pad + dd)];
      }
    });
  }
  STOP_PROFILING(&comms_profile, __func__);
}