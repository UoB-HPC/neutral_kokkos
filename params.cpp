#include "params.h"
#include "shared.h"
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

int get_parameter_line(const char* param_name, const char* filename,
                       char** param_line);

// Fetches a string parameter from the file
char* get_parameter(const char* param_name, const char* filename) {
  char* parameter = (char*)malloc(sizeof(char) * MAX_STR_LEN);
  if (!parameter) {
    TERMINATE("Could not allocate space for parameter.");
  }

  if (!get_parameter_line(param_name, filename, &parameter)) {
    TERMINATE("Could not find the parameter %s in file %s.\n", param_name,
              filename);
  }

  skip_whitespace(&parameter);
  size_t str_len = strlen(parameter);
  for (size_t ii = 0; ii < str_len; ++ii) {
    // Strip newline or trailing whitespace
    if (parameter[ii] == '\n' || parameter[ii] == '\r' ||
        parameter[ii] == ' ') {
      parameter[ii] = '\0';
    }
  }

  return parameter;
}

// Returns a parameter from the parameter file of type integer
int get_int_parameter(const char* param_name, const char* filename) {
  char line[MAX_STR_LEN];
  char* param_line = line;
  if (!get_parameter_line(param_name, filename, &param_line)) {
    TERMINATE("Could not find the parameter %s in file %s.\n", param_name,
              filename);
  }

  int value;
  sscanf(param_line, "%d", &value);
  return value;
}

// Returns a parameter from the parameter file of type double
double get_double_parameter(const char* param_name, const char* filename) {
  char line[MAX_STR_LEN];
  char* param_line = line;
  if (!get_parameter_line(param_name, filename, &param_line)) {
    TERMINATE("Could not find the parameter %s in file %s.\n", param_name,
              filename);
  }

  double value;
  sscanf(param_line, "%lf", &value);
  return value;
}

// Fetches all of the problem parameters
int get_key_value_parameter(const char* specifier, const char* filename,
                            char* keys,  Kokkos::View<double *>::HostMirror values, int* nkeys) {
  char line[MAX_STR_LEN];
  char* param_line = line;

  if (!get_parameter_line(specifier, filename, &param_line)) {
    return 0;
  }

  const size_t param_len = strlen(param_line);
  assert(param_len < MAX_STR_LEN);

  // Parse the kv pairs
  int key_index = 0;
  int parse_value = 0;
  for (size_t cc = 0; cc < param_len; ++cc) {
    if (param_line[cc] == '=') {
      // We are finished adding the key, time to get value
      keys[((*nkeys) * MAX_STR_LEN) + key_index] = '\0';
      parse_value = 1;
      key_index = 0;
    } else if (param_line[cc] == '#') {
      break; // We have encountered a comment so bail
    } else if (param_line[cc] != ' ') {
      if (parse_value) {
        sscanf(&param_line[cc], "%lf", &values[(*nkeys)++]);

        // Move the pointer to next space or end of line
        while ((++cc < param_len) && (param_line[cc] != ' '))
          ;
        parse_value = 0;
      } else {
        keys[((*nkeys) * MAX_STR_LEN) + key_index++] = param_line[cc];
      }
    }
  }

  return 1;
}

// Fetches all of the problem parameters
int get_key_value_parameter_double(const char* specifier, const char* filename,
                            char* keys,  double* values, int* nkeys) {
  char line[MAX_STR_LEN];
  char* param_line = line;

  if (!get_parameter_line(specifier, filename, &param_line)) {
    return 0;
  }

  const size_t param_len = strlen(param_line);
  assert(param_len < MAX_STR_LEN);

  // Parse the kv pairs
  int key_index = 0;
  int parse_value = 0;
  for (size_t cc = 0; cc < param_len; ++cc) {
    if (param_line[cc] == '=') {
      // We are finished adding the key, time to get value
      keys[((*nkeys) * MAX_STR_LEN) + key_index] = '\0';
      parse_value = 1;
      key_index = 0;
    } else if (param_line[cc] == '#') {
      break; // We have encountered a comment so bail
    } else if (param_line[cc] != ' ') {
      if (parse_value) {
        sscanf(&param_line[cc], "%lf", &values[(*nkeys)++]);

        // Move the pointer to next space or end of line
        while ((++cc < param_len) && (param_line[cc] != ' '))
          ;
        parse_value = 0;
      } else {
        keys[((*nkeys) * MAX_STR_LEN) + key_index++] = param_line[cc];
      }
    }
  }

  return 1;
}

// Fetches a line from a parameter file with corresponding token
int get_parameter_line(const char* param_name, const char* filename,
                       char** param_line) {
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    TERMINATE("Could not open the parameter file: %s.\n", filename);
  }

  char tok[MAX_STR_LEN];
  while (fgets(*param_line, MAX_STR_LEN, fp)) {
    skip_whitespace(param_line);

    // Read in the parameter name
    sscanf(*param_line, "%s", tok);
    if (strmatch(tok, param_name)) {
      *param_line += strlen(tok);
      skip_whitespace(param_line);
      fclose(fp);
      return 1;
    }
  }

  fclose(fp);
  return 0;
}

// Skips any leading whitespace
void skip_whitespace(char** param_line) {
  for (unsigned long ii = 0; ii < strlen(*param_line); ++ii) {
    if (isspace((*param_line)[0])) {
      (*param_line)++;
    } else {
      return;
    }
  }
}

// Reads a token from the string
void read_token(char** line, const char* format, void* var) {
  char temp[MAX_STR_LEN];
  skip_whitespace(line);
  sscanf(*line, "%s", temp);
  sscanf(*line, format, var);
  *line += strlen(temp);
}
