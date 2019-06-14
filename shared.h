#ifndef __SHAREDHDR
#define __SHAREDHDR

#pragma once

#include "profiler.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#define ARCH_PARAMS "arch.params"
#define ENABLE_VISIT_DUMPS 1 // Enables visit dumps
#define VEC_ALIGN 256 // The vector alignment to be used by memory allocators
#define TAG_VISIT0 1000
#define TAG_VISIT1 1001
#define MAX_STR_LEN 1024
#define MAX_KEYS 10
#define GB ((1024.0) * (1024.0) * (1024.0))

// Helper macros
#define strmatch(a, b) (strcmp((a), (b)) == 0)

#ifndef __cplusplus
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#define samesign(a, b) ((a * b) > 0.0)
#define absmin(a, b) ((fabs(a) < fabs(b)) ? (a) : (b))
#define minmod(a, b) (samesign((a), (b)) ? (absmin((a), (b))) : (0.0))
#define within_tolerance(a, b, eps)                                            \
  (!isnan(a) && !isnan(b) &&                                                   \
   fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * eps))
#define kronecker_delta(a, b) (((a) == (b)) ? 1 : 0)
#define triangle(a) ((a) * ((a) + 1) / 2)
#define dswap(a, b)                                                            \
  {                                                                            \
    double t = a;                                                              \
    a = b;                                                                     \
    b = t;                                                                     \
  }

#define TERMINATE(...)                                                         \
  fprintf(stderr, __VA_ARGS__);                                                \
  fprintf(stderr, " %s:%d\n", __FILE__, __LINE__);                             \
  exit(EXIT_FAILURE);

enum { RECV = 0, SEND = 1 }; // Whether data is sent to/received from device

// Global profile hooks
extern struct Profile compute_profile;
extern struct Profile comms_profile;

#ifdef __cplusplus
extern "C" {
#endif

// Initialises devices in implementation-specific manner
void initialise_devices(int rank);

// Allocation and deallocation routines (these need templating away)
size_t allocate_data(Kokkos::View<double*>* buf, size_t len);
size_t allocate_float_data(Kokkos::View<float*>* buf, size_t len);
size_t allocate_int_data(Kokkos::View<int*>* buf, size_t len);
size_t allocate_uint64_data(Kokkos::View<uint64_t*>* buf, const size_t len);
// size_t allocate_complex_double_data(_Complex double** buf, const size_t len);

void allocate_host_data(Kokkos::View<double*>::HostMirror* buf, size_t len);
void allocate_host_float_data(Kokkos::View<float*>::HostMirror* buf, size_t len);
void allocate_host_int_data(Kokkos::View<int*>::HostMirror* buf, size_t len);
void allocate_host_uint64_data(Kokkos::View<uint64_t*>::HostMirror* buf, size_t len);
// void allocate_host_complex_double_data(_Complex double** buf, size_t len);

void deallocate_data(Kokkos::View<double*> buf);
void deallocate_float_data(Kokkos::View<float*> buf);
void deallocate_int_data(Kokkos::View<int*> buf);
void deallocate_uint64_t_data(Kokkos::View<uint64_t*> buf);
// void deallocate_complex_double_data(_Complex double* buf);

void deallocate_host_data(Kokkos::View<double*>::HostMirror buf);
void deallocate_host_float_data(Kokkos::View<float*>::HostMirror buf);
void deallocate_host_int_data(Kokkos::View<int*>::HostMirror buf);
void deallocate_host_uint64_t_data(Kokkos::View<uint64_t*>::HostMirror buf);
// void deallocate_host_complex_double_data(_Complex double* buf);

void copy_buffer_SEND(const size_t len, Kokkos::View<double*>::HostMirror* src, Kokkos::View<double*>* dst);
void copy_float_buffer_SEND(const size_t len, Kokkos::View<float*>::HostMirror* src, Kokkos::View<float*>* dst);
void copy_int_buffer_SEND(const size_t len, Kokkos::View<int*>::HostMirror* src, Kokkos::View<int*>* dst);

void copy_buffer_RECEIVE(const size_t len, Kokkos::View<double*>* src, Kokkos::View<double*>::HostMirror* dst);
// void copy_buffer(const size_t len, double** src, double** dst, int send);
// void copy_float_buffer(const size_t len, float** src, float** dst, int send);
// void copy_int_buffer(const size_t len, int** src, int** dst, int send);
// void copy_uint64_buffer(const size_t len, uint64_t** src, uint64_t** dst,
//                         int send);
void move_host_buffer_to_device(const size_t len, Kokkos::View<double*>::HostMirror* src, Kokkos::View<double*>* dst);
// void move_host_float_buffer_to_device(const size_t len, float** src, float** dst);

// Write out data for visualisation in visit
void write_to_visit(const int nx, const int ny, const int x_off,
                    const int y_off, const double* data, const char* name,
                    const int step, const double time);
void write_to_visit_3d(const int nx, const int ny, const int nz,
                       const int x_off, const int y_off, const int z_off,
                       const double* data, const char* name, const int step,
                       const double time);

// Collects all of the mesh data from the fleet of ranks and then writes to
// visit
void write_all_ranks_to_visit(const int global_nx, const int global_ny,
                              const int local_nx, const int local_ny,
                              const int pad, const int x_off, const int y_off,
                              const int rank, const int nranks, int* neighbours,
                              double* local_arr, const char* name, const int tt,
                              const double elapsed_sim_time);

#ifdef __cplusplus
}
#endif

#endif
