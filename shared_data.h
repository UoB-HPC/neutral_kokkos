#ifndef __SHAREDDATAHDR
#define __SHAREDDATAHDR

#include "mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: MAKE IT SO THAT shared_data IS LOCAL TO THE APPLICATIONS???

// Contains all of the shared_data information for the solver
typedef struct {
  // Shared shared_data (share data)
  Kokkos::View<double *> density; // Density
  Kokkos::View<double *> energy;  // Energy

  // Paired shared_data (share capacity)
  Kokkos::View<double *> Ap;          // HOT: Coefficient matrix A, by conjugate vector p
  Kokkos::View<double *> density_old; // FLOW: Density at beginning of timestep

  Kokkos::View<double *> s_x; // HOT: Coefficients in temperature direction
  Kokkos::View<double *> Qxx; // FLOW: Artificial viscous term in temperature direction

  Kokkos::View<double *> s_y; // HOT: Coefficients in y direction
  Kokkos::View<double *> Qyy; // FLOW: Artificial viscous term in y direction

  Kokkos::View<double *> s_z; // HOT: Coefficients in z direction
  Kokkos::View<double *> Qzz; // FLOW: Artificial viscous term in z direction

  Kokkos::View<double *> r;        // HOT: The residual vector
  Kokkos::View<double *> pressure; // FLOW: The pressure

  Kokkos::View<double *> temperature; // HOT: The solution vector (new energy)
  Kokkos::View<double *> u;           // FLOW: The velocity in the temperature direction

  Kokkos::View<double *> p; // HOT: The conjugate vector
  Kokkos::View<double *> v; // FLOW: The velocity in the y direction

  Kokkos::View<double *> reduce_array0;
  Kokkos::View<double *> reduce_array1;

} SharedData;

// Initialises the shared_data variables
void initialise_shared_data_2d(const int local_nx, const int local_ny,
                               const int pad, const double mesh_width,
                               const double mesh_height,
                               const char* problem_def_filename,
                               const Kokkos::View<double *> edgex,
                               const Kokkos::View<double *> edgey,
                               SharedData* shared_data);

// Initialise state data in device specific manner
void set_problem_2d(const int local_nx, const int local_ny, const int pad,
                    const double mesh_width, const double mesh_height,
                    const Kokkos::View<double *> edgex,
                    const Kokkos::View<double *> edgey,
                    const int ndims,
                    const char* problem_def_filename,
                    const Kokkos::View<double *> density,
                    const Kokkos::View<double *> energy,
                    const Kokkos::View<double *> temperature);

// // Initialises the shared_data variables
// void initialise_shared_data_3d(
//     const int local_nx, const int local_ny, const int local_nz, const int pad,
//     const double mesh_width, const double mesh_height, const double mesh_depth,
//     const char* problem_def_filename, const double* edgex, const double* edgey,
//     const double* edgez, SharedData* shared_data);

// void set_problem_3d(const int local_nx, const int local_ny, const int local_nz,
//                     const int pad, const double mesh_width,
//                     const double mesh_height, const double mesh_depth,
//                     const double* edgex, const double* edgey,
//                     const double* edgez, const int ndims,
//                     const char* problem_def_filename, double* density,
//                     double* energy, double* temperature);

// // Deallocate all of the shared_data memory
// void finalise_shared_data(SharedData* shared_data);

#ifdef __cplusplus
}
#endif

#endif
