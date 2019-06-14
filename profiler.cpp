#include "profiler.h"
#include "shared.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Internally start the profiling timer
void profiler_start_timer(struct Profile* profile) {
#ifndef TIME_UTC
  clock_gettime(CLOCK_MONOTONIC, &profile->profiler_start);
#else
  timespec_get(&profile->profiler_start, TIME_UTC);
#endif
}

// Internally end the profiling timer and store results
void profiler_end_timer(struct Profile* profile, const char* entry_name) {
#ifndef TIME_UTC
  clock_gettime(CLOCK_MONOTONIC, &profile->profiler_end);
#else
  timespec_get(&profile->profiler_end, TIME_UTC);
#endif

  // Check if an entry exists
  int ii;
  for (ii = 0; ii < profile->profiler_entry_count; ++ii) {
    if (strmatch(profile->profiler_entries[ii].name, entry_name)) {
      break;
    }
  }

  // Don't overrun
  if (ii >= PROFILER_MAX_ENTRIES) {
    TERMINATE("Attempted to profile too many entries, maximum is %d\n",
              PROFILER_MAX_ENTRIES);
  }

  // Create new entry
  if (ii == profile->profiler_entry_count) {
    profile->profiler_entry_count++;
    strcpy(profile->profiler_entries[ii].name, entry_name);
    profile->profiler_entries[ii].time = 0.0;
  }

// Update number of calls and time
  double elapsed =
      (profile->profiler_end.tv_sec - profile->profiler_start.tv_sec) +
      (profile->profiler_end.tv_nsec - profile->profiler_start.tv_nsec) *
          1.0E-9;

  profile->profiler_entries[ii].time += elapsed;
  profile->profiler_entries[ii].calls++;
}

// Print the profiling results to output
void profiler_print_full_profile(struct Profile* profile) {
  printf("\n-------------------------------------------------------------\n");
  printf("\nProfiling Results:\n\n");
  printf("%-39s%8s%28s\n", "Kernel Name", "Calls", "Runtime (s)");

  double total_elapsed_time = 0.0;
  for (int ii = 0; ii < profile->profiler_entry_count; ++ii) {
    total_elapsed_time += profile->profiler_entries[ii].time;
    printf("%-39s%8d%28.9F\n", profile->profiler_entries[ii].name,
           profile->profiler_entries[ii].calls,
           profile->profiler_entries[ii].time);
  }

  printf("\nTotal elapsed time: %.9Fs, entries * are excluded.\n",
         total_elapsed_time);
  printf("\n-------------------------------------------------------------\n\n");
}

// Prints profile without extra details
void profiler_print_simple_profile(struct Profile* profile) {
  for (int ii = 0; ii < profile->profiler_entry_count; ++ii) {
    printf("\033[1m\033[30m%s\033[0m: %.8lfs (%d calls)\n",
           profile->profiler_entries[ii].name,
           profile->profiler_entries[ii].time,
           profile->profiler_entries[ii].calls);
  }
}

// Gets an individual profile entry
struct ProfileEntry profiler_get_profile_entry(struct Profile* profile,
                                               const char* entry_name) {

  for (int ii = 0; ii < profile->profiler_entry_count; ++ii) {
    if (strmatch(profile->profiler_entries[ii].name, entry_name)) {
      return profile->profiler_entries[ii];
    }
  }

  TERMINATE("Attempted to retrieve missing profile entry %s\n", entry_name);
}

double profiler_get_time(struct Profile* profile, const char* entry_name) {

  for (int ii = 0; ii < profile->profiler_entry_count; ++ii) {
    if (strmatch(profile->profiler_entries[ii].name, entry_name)) {
      return profile->profiler_entries[ii].time;
    }
  }

  TERMINATE("Attempted to retrieve missing profile entry %s\n", entry_name);
}

void profiler_init(struct Profile* profile) {
  for(int i = 0; i < PROFILER_MAX_ENTRIES; ++i) {
    profile->profiler_entries[i].time = 0.0;
  }
}
