#include "mesh.h"
#include "params.h"
#include "shared.h"
#include <assert.h>
#include <stdlib.h>

// Initialise the mesh describing variables
void initialise_mesh_2d(Mesh* mesh) {
  allocate_data(&mesh->edgex, (mesh->local_nx + 1));
  allocate_data(&mesh->edgey, (mesh->local_ny + 1));
  allocate_data(&mesh->edgedx, (mesh->local_nx + 1));
  allocate_data(&mesh->edgedy, (mesh->local_ny + 1));
  allocate_data(&mesh->celldx, (mesh->local_nx + 1));
  allocate_data(&mesh->celldy, (mesh->local_ny + 1));

  mesh_data_init_2d(mesh->local_nx, mesh->local_ny, mesh->global_nx,
                    mesh->global_ny, mesh->pad, mesh->x_off, mesh->y_off,
                    mesh->width, mesh->height, mesh->edgex, mesh->edgey,
                    mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  allocate_data(&mesh->north_buffer_out, (mesh->local_nx + 1) * mesh->pad);
  allocate_data(&mesh->east_buffer_out, (mesh->local_ny + 1) * mesh->pad);
  allocate_data(&mesh->south_buffer_out, (mesh->local_nx + 1) * mesh->pad);
  allocate_data(&mesh->west_buffer_out, (mesh->local_ny + 1) * mesh->pad);
  allocate_data(&mesh->north_buffer_in, (mesh->local_nx + 1) * mesh->pad);
  allocate_data(&mesh->east_buffer_in, (mesh->local_ny + 1) * mesh->pad);
  allocate_data(&mesh->south_buffer_in, (mesh->local_nx + 1) * mesh->pad);
  allocate_data(&mesh->west_buffer_in, (mesh->local_ny + 1) * mesh->pad);

  allocate_host_data(&mesh->h_north_buffer_out,
                     (mesh->local_nx + 1) * mesh->pad);
  allocate_host_data(&mesh->h_east_buffer_out,
                     (mesh->local_ny + 1) * mesh->pad);
  allocate_host_data(&mesh->h_south_buffer_out,
                     (mesh->local_nx + 1) * mesh->pad);
  allocate_host_data(&mesh->h_west_buffer_out,
                     (mesh->local_ny + 1) * mesh->pad);
  allocate_host_data(&mesh->h_north_buffer_in,
                     (mesh->local_nx + 1) * mesh->pad);
  allocate_host_data(&mesh->h_east_buffer_in, (mesh->local_ny + 1) * mesh->pad);
  allocate_host_data(&mesh->h_south_buffer_in,
                     (mesh->local_nx + 1) * mesh->pad);
  allocate_host_data(&mesh->h_west_buffer_in, (mesh->local_ny + 1) * mesh->pad);
}

// // Initialise the mesh describing variables
// void initialise_mesh_3d(Mesh* mesh) {
//   allocate_data(&mesh->edgex, (mesh->local_nx + 1));
//   allocate_data(&mesh->edgey, (mesh->local_ny + 1));
//   allocate_data(&mesh->edgez, (mesh->local_nz + 1));
//   allocate_data(&mesh->edgedx, (mesh->local_nx + 1));
//   allocate_data(&mesh->edgedy, (mesh->local_ny + 1));
//   allocate_data(&mesh->edgedz, (mesh->local_nz + 1));
//   allocate_data(&mesh->celldx, (mesh->local_nx + 1));
//   allocate_data(&mesh->celldy, (mesh->local_ny + 1));
//   allocate_data(&mesh->celldz, (mesh->local_nz + 1));

//   mesh_data_init_3d(mesh->local_nx, mesh->local_ny, mesh->local_nz,
//                     mesh->global_nx, mesh->global_ny, mesh->local_nz, mesh->pad,
//                     mesh->x_off, mesh->y_off, mesh->z_off, mesh->width,
//                     mesh->height, mesh->depth, mesh->edgex, mesh->edgey,
//                     mesh->edgez, mesh->edgedx, mesh->edgedy, mesh->edgedz,
//                     mesh->celldx, mesh->celldy, mesh->celldz);

//   allocate_data(&mesh->north_buffer_out,
//                 (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->east_buffer_out,
//                 (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->south_buffer_out,
//                 (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->west_buffer_out,
//                 (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->front_buffer_out,
//                 (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);
//   allocate_data(&mesh->back_buffer_out,
//                 (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);
//   allocate_data(&mesh->north_buffer_in,
//                 (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->east_buffer_in,
//                 (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->south_buffer_in,
//                 (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->west_buffer_in,
//                 (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_data(&mesh->front_buffer_in,
//                 (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);
//   allocate_data(&mesh->back_buffer_in,
//                 (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);

//   allocate_host_data(&mesh->h_north_buffer_out,
//                      (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_east_buffer_out,
//                      (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_south_buffer_out,
//                      (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_west_buffer_out,
//                      (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_front_buffer_out,
//                      (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_back_buffer_out,
//                      (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_north_buffer_in,
//                      (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_east_buffer_in,
//                      (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_south_buffer_in,
//                      (mesh->local_nx + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_west_buffer_in,
//                      (mesh->local_ny + 1) * (mesh->local_nz + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_front_buffer_in,
//                      (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);
//   allocate_host_data(&mesh->h_back_buffer_in,
//                      (mesh->local_nx + 1) * (mesh->local_ny + 1) * mesh->pad);
// }

// Deallocate all of the mesh memory
void finalise_mesh(Mesh* mesh) {
  deallocate_data(mesh->edgedy);
  deallocate_data(mesh->celldy);
  deallocate_data(mesh->edgedx);
  deallocate_data(mesh->celldx);
  deallocate_data(mesh->north_buffer_out);
  deallocate_data(mesh->east_buffer_out);
  deallocate_data(mesh->south_buffer_out);
  deallocate_data(mesh->west_buffer_out);
  deallocate_data(mesh->north_buffer_in);
  deallocate_data(mesh->east_buffer_in);
  deallocate_data(mesh->south_buffer_in);
  deallocate_data(mesh->west_buffer_in);
}
