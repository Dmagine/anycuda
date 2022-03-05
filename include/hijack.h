/*
 * Tencent is pleased to support the open source community by making TKEStack
 * available.
 *
 * Copyright (C) 2012-2019 Tencent. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations under the License.
 */

#ifndef HIJACK_LIBRARY_H
#define HIJACK_LIBRARY_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nvml-subset.h"
#include "cuda-subset.h"

/**
 * Controller configuration base path
 */
#define ANYCUDA_CONFIG_PATH "/var/run/anylearn"

/**
 * Controller pid information file name
 */
#define PIDS_CONFIG_NAME "pids.config"

/**
 * Controller configuration file name
 */
#define PIDS_CONFIG_PATH (ANYCUDA_CONFIG_PATH "/" PIDS_CONFIG_NAME)

/**
 * Default prefix for cgroup path
 */
#define EMPTY_PREFIX "0xdead"

/**
 * Proc file path for driver version
 */
#define DRIVER_VERSION_PROC_PATH "/proc/driver/nvidia/version"

/**
 * Driver regular expression pattern
 */
#define DRIVER_VERSION_MATCH_PATTERN "([0-9]+)(\\.[0-9]+)+"

/**
 * Max sample pid size
 */
#define MAX_PIDS (1024)

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define ROUND_UP(n, base) ((n) % (base) ? (n) + (base) - (n) % (base) : (n))

#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2 * !!(condition)]))

#define CAS(ptr, old, new) __sync_bool_compare_and_swap((ptr), (old), (new))
#define UNUSED __attribute__((unused))

#define MILLISEC (1000UL * 1000UL)

#define TIME_TICK (10)
#define FACTOR (32)
#define MAX_UTILIZATION (100)
#define CHANGE_LIMIT_INTERVAL (30)
#define USAGE_THRESHOLD (5)

#define GET_VALID_VALUE(x) (((x) >= 0 && (x) <= 100) ? (x) : 0)
#define CODEC_NORMALIZE(x) (x * 85 / 100)

  typedef struct
  {
    void *fn_ptr;
    char *name;
  } entry_t;

  typedef struct
  {
    int major;
    int minor;
  } __attribute__((packed, aligned(8))) version_t;

  typedef struct
  {
    char *uuid;
    int limit;
  } __attribute__((packed, aligned(8))) deviceLimit;

  /**
   * Podconf data format
   */
  typedef struct
  {
    char pod_name[48];
    char resource_name[48];
    char *gpu_uuids[16];
    char gpu_cuda_uuids[16][16];
    int gpu_count;

    int gpu_mem_limit_valid;
    size_t gpu_mem_limit[16];

    int valid;
  } __attribute__((packed, aligned(8))) resource_data_t;

  typedef struct
  {
    CUdevice device;
    CUuuid uuid;
  } __attribute__((packed, aligned(8))) device_info;

  typedef enum
  {
    INFO = 0,
    ERROR = 1,
    WARNING = 2,
    FATAL = 3,
    VERBOSE = 4,
  } log_level_enum_t;

#define LOGGER(level, format, ...)                              \
  ({                                                            \
    char *_print_level_str = getenv("LOGGER_LEVEL");            \
    int _print_level = 3;                                       \
    if (_print_level_str)                                       \
    {                                                           \
      _print_level = (int)strtoul(_print_level_str, NULL, 10);  \
      _print_level = _print_level < 0 ? 3 : _print_level;       \
    }                                                           \
    if (level <= _print_level)                                  \
    {                                                           \
      fprintf(stderr, "%s:%d " format "\n", __FILE__, __LINE__, \
              ##__VA_ARGS__);                                   \
    }                                                           \
    if (level == FATAL)                                         \
    {                                                           \
      exit(-1);                                                 \
    }                                                           \
  })

  /**
   * Read controller configuration from POD_CONF
   *
   * @return 0 -> success
   */
  int read_anylearn_podconf();

  /**
   * Load library and initialize some data
   */
  void load_necessary_data();

#ifdef __cplusplus
}
#endif

#endif
