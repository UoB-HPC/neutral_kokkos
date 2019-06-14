#include "mesh.h"
#include "params.h"
#include "shared.h"
#include "shared_data.h"
#include "shared.h"
#include <math.h>
#include <stdlib.h>

// Allocates a double precision array
size_t allocate_data(Kokkos::View<double*>* buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    new(buf) Kokkos::View<double*>("device", len);

    Kokkos::View<double*> local_buf(*buf);
    Kokkos::parallel_for(len, KOKKOS_LAMBDA (int i) {
        local_buf[i] = 0.0;
    });
    Kokkos::fence();
    
    return sizeof(double) * len;
}

size_t allocate_float_data(Kokkos::View<float*>* buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    new(buf) Kokkos::View<float*>("device", len);

    Kokkos::View<float*> local_buf(*buf);
    Kokkos::parallel_for(len, KOKKOS_LAMBDA (int i) {
        local_buf[i] = 0.0f;
    });
    Kokkos::fence();
    
    return sizeof(float) * len;
}


size_t allocate_int_data(Kokkos::View<int*>* buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    new(buf) Kokkos::View<int*>("device", len);

    Kokkos::View<int*> local_buf(*buf);
    Kokkos::parallel_for(len, KOKKOS_LAMBDA (int i) {
        local_buf[i] = 0;
    });
    Kokkos::fence();
    
    return sizeof(int) * len;
}

size_t allocate_uint64_data(Kokkos::View<uint64_t*>* buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    new(buf) Kokkos::View<uint64_t*>("device", len);

    Kokkos::View<uint64_t*> local_buf(*buf);
    Kokkos::parallel_for(len, KOKKOS_LAMBDA (int i) {
        local_buf[i] = 0;
    });
    Kokkos::fence();
    
    return sizeof(uint64_t) * len;
}

// Deallocate a double array
void deallocate_data(Kokkos::View<double*> buf) {
    //Do nothing
}

// Deallocate a float array
void deallocate_float_data(Kokkos::View<float*> buf) {
    //Do nothing
}

// Deallocate an int array
void deallocate_int_data(Kokkos::View<int*> buf) {
    //Do nothing
}

// Deallocate an uint_64t array
void deallocate_uint64_t_data(Kokkos::View<uint64_t*> buf) {
    //Do nothing
}

// Allocates some double precision data
void allocate_host_data(Kokkos::View<double*>::HostMirror* buf, const size_t len) {
    if(len == 0) {
        return;
    }

    new(buf) Kokkos::View<double*>::HostMirror("host", len);
    
    for (size_t ii = 0; ii < len; ++ii) {
        (*buf)[ii] = 1.0;
    }
}

// Allocates some single precision data
void allocate_host_float_data(Kokkos::View<float*>::HostMirror* buf, const size_t len) {
    if(len == 0) {
        return;
    }

    new(buf) Kokkos::View<float*>::HostMirror("host", len);
    
    for (size_t ii = 0; ii < len; ++ii) {
        (*buf)[ii] = 0.0f;
    }
}

void allocate_host_int_data(Kokkos::View<int*>::HostMirror* buf, const size_t len) {
    if(len == 0) {
        return;
    }

    new(buf) Kokkos::View<int*>::HostMirror("host", len);
    
    for (size_t ii = 0; ii < len; ++ii) {
        (*buf)[ii] = 0;
    }
}

void allocate_host_uint64_t_data(Kokkos::View<uint64_t*>::HostMirror* buf, const size_t len) {
    if(len == 0) {
        return;
    }

    new(buf) Kokkos::View<uint64_t*>::HostMirror("host", len);
    
    for (size_t ii = 0; ii < len; ++ii) {
        (*buf)[ii] = 0;
    }
}

// Deallocates a data array
void deallocate_host_data(Kokkos::View<double*>::HostMirror buf) {
    //Do nothing
}

void deallocate_host_float_data(Kokkos::View<float*>::HostMirror buf) {
    //Do nothing
}

void deallocate_host_int_data(Kokkos::View<int*>::HostMirror buf) {
    //Do nothing
}

void deallocate_host_uint64_t_data(Kokkos::View<uint64_t*>::HostMirror buf) {
    //Do nothing
}

// Initialises mesh data in device specific manner
void mesh_data_init_2d(const int local_nx, const int local_ny,
    const int global_nx, const int global_ny, const int pad,
    const int x_off, const int y_off, const double width,
    const double height, Kokkos::View<double *> edgex, Kokkos::View<double *> edgey,
    Kokkos::View<double *> edgedx, Kokkos::View<double *> edgedy, Kokkos::View<double *> celldx,
    Kokkos::View<double *> celldy) {

    // Simple uniform rectilinear initialisation
    Kokkos::parallel_for(local_nx+1, KOKKOS_LAMBDA (const int ii)
    {
        edgedx[ii] = width / (global_nx);

        // Note: correcting for padding
        edgex[ii] = edgedx[ii] * (x_off + ii - pad);
    });


    Kokkos::parallel_for(local_nx, KOKKOS_LAMBDA (const int ii)
    {
        celldx[ii] = width / (global_nx);
    });


    Kokkos::parallel_for(local_ny+1, KOKKOS_LAMBDA (const int ii)
    {
        edgedy[ii] = height / (global_ny);

        // Note: correcting for padding
        edgey[ii] = edgedy[ii] * (y_off + ii - pad);
    });


    Kokkos::parallel_for(local_ny, KOKKOS_LAMBDA (const int ii)
    {
        celldy[ii] = height / (global_ny);
    });
    Kokkos::fence();
}

void copy_buffer_SEND(const size_t len, Kokkos::View<double*>::HostMirror* src, Kokkos::View<double*>* dst) {
    deep_copy(*dst, *src);
}

void copy_float_buffer_SEND(const size_t len, Kokkos::View<float*>::HostMirror* src, Kokkos::View<float*>* dst) {
    deep_copy(*dst, *src);
}

void copy_int_buffer_SEND(const size_t len, Kokkos::View<int*>::HostMirror* src, Kokkos::View<int*>* dst) {
    deep_copy(*dst, *src);
}

void copy_buffer_RECEIVE(const size_t len, Kokkos::View<double*>* src, Kokkos::View<double*>::HostMirror* dst) {
    deep_copy(*dst, *src);
}

void move_host_buffer_to_device(const size_t len, Kokkos::View<double*>::HostMirror* src, Kokkos::View<double*>* dst) {
  allocate_data(dst, len);
  copy_buffer_SEND(len, src, dst);
  deallocate_host_data(*src);
}

// Initialise state data in device specific manner
void set_problem_2d(const int local_nx, const int local_ny, const int pad,
                    const double mesh_width, const double mesh_height,
                    const Kokkos::View<double *> edgex,
                    const Kokkos::View<double *> edgey,
                    const int ndims,
                    const char* problem_def_filename,
                    Kokkos::View<double *> density,
                    Kokkos::View<double *> energy,
                    Kokkos::View<double *> temperature) {

    Kokkos::View<int *>::HostMirror h_keys;
    Kokkos::View<int *> d_keys;
    allocate_int_data(&d_keys, MAX_KEYS);
    allocate_host_int_data(&h_keys, MAX_KEYS);

    Kokkos::View<double *>::HostMirror h_values;
    Kokkos::View<double *> d_values;
    allocate_data(&d_values, MAX_KEYS);
    allocate_host_data(&h_values, MAX_KEYS);

    int nentries = 0;
    while (1) {
        char specifier[MAX_STR_LEN];
        char keys[MAX_STR_LEN * MAX_KEYS];
        sprintf(specifier, "problem_%d", nentries++);

        int nkeys = 0;
    if (!get_key_value_parameter(specifier, problem_def_filename, keys,
          h_values, &nkeys)) {
      break;
    }

    // The last four keys are the bound specification
    double xpos = h_values[nkeys - 4] * mesh_width;
    double ypos = h_values[nkeys - 3] * mesh_height;
    double width = h_values[nkeys - 2] * mesh_width;
    double height = h_values[nkeys - 1] * mesh_height;

    for (int kk = 0; kk < nkeys - (2 * ndims); ++kk) {
      const char* key = &keys[kk * MAX_STR_LEN];
      if (strmatch(key, "density")) {
        h_keys[kk] = DENSITY_KEY;
      } else if (strmatch(key, "energy")) {
        h_keys[kk] = ENERGY_KEY;
      } else if (strmatch(key, "temperature")) {
        h_keys[kk] = TEMPERATURE_KEY;
      } else {
        TERMINATE("Found unrecognised key in %s : %s.\n", problem_def_filename,
            key);
      }
    }

    copy_int_buffer_SEND(MAX_KEYS, &h_keys, &d_keys);
    copy_buffer_SEND(MAX_KEYS, &h_values, &d_values);

    Kokkos::parallel_for(local_nx*local_ny, KOKKOS_LAMBDA (const int i)
    {
        const int ii = i / local_nx;
        const int jj = i % local_nx;
        double global_xpos = edgex[jj];
        double global_ypos = edgey[ii];

        // Check we are in bounds of the problem entry
        if (global_xpos >= xpos && 
            global_ypos >= ypos && 
            global_xpos < xpos + width && 
            global_ypos < ypos + height) {

            // The upper bound excludes the bounding box for the entry
            for (int nn = 0; nn < nkeys - (2 * ndims); ++nn) {
                const int key = d_keys[nn];
                if (key == DENSITY_KEY) {
                    density[i] = d_values[nn];
                } else if (key == ENERGY_KEY) {
                    energy[i] = d_values[nn];
                } else if (key == TEMPERATURE_KEY) {
                    temperature[i] = d_values[nn];
                }
            }
        }
    });

  }

  deallocate_host_int_data(h_keys);
  deallocate_host_data(h_values);
}
