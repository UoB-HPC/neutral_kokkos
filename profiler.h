#ifndef __PROFILERH
#define __PROFILERH

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

/*
 *		PROFILING TOOL
 *		Not thread safe.
 */

#define PROFILER_MAX_NAME 128
#define PROFILER_MAX_ENTRIES 1024

#ifdef __cplusplus
extern "C" {
#endif

struct ProfileEntry {
  int calls;
  double time;
  char name[PROFILER_MAX_NAME];
};

struct Profile {
#ifdef __APPLE__
  uint64_t profiler_start;
  uint64_t profiler_end;
#else
  struct timespec profiler_start;
  struct timespec profiler_end;
#endif

  int profiler_entry_count;
  struct ProfileEntry profiler_entries[PROFILER_MAX_ENTRIES];
};

void profiler_start_timer(struct Profile* profile);
void profiler_end_timer(struct Profile* profile, const char* entry_name);
void profiler_print_simple_profile(struct Profile* profile);
void profiler_print_full_profile(struct Profile* profile);
struct ProfileEntry profiler_get_profile_entry(struct Profile* profile,
                                               const char* entry_name);
double profiler_get_time(struct Profile* profile, const char* entry_name);
void profiler_init(struct Profile* profile);

// Allows compile-time optimised conditional profiling
#ifdef ENABLE_PROFILING

#define START_PROFILING(profile) profiler_start_timer(profile)

#define STOP_PROFILING(profile, name) profiler_end_timer(profile, name)

#define PRINT_PROFILING_RESULTS(profile) profiler_print_full_profile(profile)

#else

#define START_PROFILING(profile) ;

#define STOP_PROFILING(profile, name) ;

#define PRINT_PROFILING_RESULTS(profile) ;

#endif

#ifdef __cplusplus
}
#endif

#endif
