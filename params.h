#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

enum { ENERGY_KEY, DENSITY_KEY, TEMPERATURE_KEY };

// Fetches a string parameter from the file
char* get_parameter(const char* param_name, const char* filename);

// Returns a parameter from the parameter file of type integer
int get_int_parameter(const char* param_name, const char* filename);

// Returns a parameter from the parameter file of type double
double get_double_parameter(const char* param_name, const char* filename);

// Skips any leading whitespace
void skip_whitespace(char** line);

// Reads a token from an input string
void read_token(char** line, const char* format, void* var);

#ifdef __cplusplus
extern "C" {
#endif

// Fetches all of the problem parameters
int get_key_value_parameter(const char* specifier, const char* filename,
                            char* keys,  Kokkos::View<double *>::HostMirror values, int* nkeys);

// Fetches all of the problem parameters
int get_key_value_parameter_double(const char* specifier, const char* filename,
                            char* keys,  double* values, int* nkeys);

#ifdef __cplusplus
}
#endif