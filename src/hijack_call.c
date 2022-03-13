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

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>

#include "include/cuda-helper.h"
#include "include/hijack.h"
#include "include/nvml-helper.h"
#include "include/cJSON.h"

extern resource_data_t g_anycuda_config;
extern cJSON *g_podconf;
extern char config_path[FILENAME_MAX];
extern device_info g_devices_info[16];
extern int g_device_count;

extern entry_t cuda_library_entry[];
extern entry_t nvml_library_entry[];

typedef void (*atomic_fn_ptr)(int, void *);

static pthread_once_t g_init_set = PTHREAD_ONCE_INIT;

static const struct timespec g_wait = {
    .tv_sec = 0,
    .tv_nsec = 1000 * MILLISEC,
};

/** internal function definition */
static void active_podconf_notifier();

static void *podconf_watcher(void *);

int read_anylearn_podconf();

static void get_used_gpu_memory(void *, CUdevice);

static void initialization();

static const char *cuda_error(CUresult, const char **);

void get_uuid_str(char *dest, CUuuid *src);

/** export function definition */
CUresult cuDriverGetVersion(int *driverVersion);
CUresult cuInit(unsigned int flag);
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes);
CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes);
CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
CUresult cuMemGetInfo(size_t *free, size_t *total);
CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra);
CUresult cuLaunch(CUfunction f);
CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams);
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void **kernelParams);
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream);
CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);

entry_t cuda_hooks_entry[] = {
    {.name = "cuDriverGetVersion", .fn_ptr = cuDriverGetVersion},
    {.name = "cuInit", .fn_ptr = cuInit},
    {.name = "cuMemAllocManaged", .fn_ptr = cuMemAllocManaged},
    {.name = "cuMemAlloc_v2", .fn_ptr = cuMemAlloc_v2},
    {.name = "cuMemAlloc", .fn_ptr = cuMemAlloc},
    {.name = "cuMemAllocPitch_v2", .fn_ptr = cuMemAllocPitch_v2},
    {.name = "cuMemAllocPitch", .fn_ptr = cuMemAllocPitch},
    {.name = "cuArrayCreate_v2", .fn_ptr = cuArrayCreate_v2},
    {.name = "cuArrayCreate", .fn_ptr = cuArrayCreate},
    {.name = "cuArray3DCreate_v2", .fn_ptr = cuArray3DCreate_v2},
    {.name = "cuArray3DCreate", .fn_ptr = cuArray3DCreate},
    {.name = "cuMipmappedArrayCreate", .fn_ptr = cuMipmappedArrayCreate},
    {.name = "cuDeviceTotalMem_v2", .fn_ptr = cuDeviceTotalMem_v2},
    {.name = "cuDeviceTotalMem", .fn_ptr = cuDeviceTotalMem},
    {.name = "cuMemGetInfo_v2", .fn_ptr = cuMemGetInfo_v2},
    {.name = "cuMemGetInfo", .fn_ptr = cuMemGetInfo},
    {.name = "cuLaunchKernel_ptsz", .fn_ptr = cuLaunchKernel_ptsz},
    {.name = "cuLaunchKernel", .fn_ptr = cuLaunchKernel},
    {.name = "cuLaunch", .fn_ptr = cuLaunch},
    {.name = "cuLaunchCooperativeKernel_ptsz",
     .fn_ptr = cuLaunchCooperativeKernel_ptsz},
    {.name = "cuLaunchCooperativeKernel", .fn_ptr = cuLaunchCooperativeKernel},
    {.name = "cuLaunchGrid", .fn_ptr = cuLaunchGrid},
    {.name = "cuLaunchGridAsync", .fn_ptr = cuLaunchGridAsync},
    {.name = "cuFuncSetBlockShape", .fn_ptr = cuFuncSetBlockShape},
};

const int cuda_hook_nums =
    sizeof(cuda_hooks_entry) / sizeof(cuda_hooks_entry[0]);

/** dynamic rate control */
typedef struct
{
  int user_current;
  int sys_current;
  int valid;
  uint64_t checktime;
  int sys_process_num;
} utilization_t;

int strsplit(const char *s, char **dest, const char *sep)
{
  char *token;
  int index = 0;
  char *src = (char *)malloc(strlen(s) + 1);
  strcpy(src, s);
  token = strtok(src, sep);
  while (token != NULL)
  {
    dest[index] = token;
    index += 1;
    token = strtok(NULL, sep);
  }
  return index;
}

const char *cuda_error(CUresult code, const char **p)
{
  CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, code, p);

  return *p;
}

static void active_podconf_notifier()
{
  pthread_t tid;

  pthread_create(&tid, NULL, podconf_watcher, NULL);
  pthread_setname_np(tid, "podconf_watcher");
}

static void *podconf_watcher(void *arg UNUSED)
{
  LOGGER(5, "start %s", __FUNCTION__);
  while (1)
  {
    nanosleep(&g_wait, NULL);
    do
    {
      read_anylearn_podconf();
    } while (!g_anycuda_config.valid);
  }
}

int read_anylearn_podconf()
{
  int fd = 0;
  int ret = 1;

  fd = open(config_path, O_RDONLY);
  if (unlikely(fd == -1))
  {
    LOGGER(VERBOSE, "can't open %s, error %s", config_path, strerror(errno));
    goto DONE;
  }
  // read podconf from json
  char buff[4096] = {"\0"};
  read(fd, buff, 4096);

  g_podconf = cJSON_Parse(buff);
  strncpy(g_anycuda_config.resource_name, cJSON_GetObjectItem(g_podconf, "resourceName")->valuestring, 48);
  char *devices = cJSON_GetObjectItem(g_podconf, "devices")->valuestring;
  g_anycuda_config.gpu_count = strsplit(devices, g_anycuda_config.gpu_uuids, ",");
  cJSON *gpu_limits = cJSON_GetObjectItem(g_podconf, "gpuLimit");
  size_t gpu_mem_limit[16];
  if (gpu_limits != NULL)
  {
    for (int i = 0; i < g_device_count; i++)
    {
      gpu_mem_limit[i] = -1;
    }
    for (int i = 0; i < g_device_count; i++)
    {
      char uuid_str[48];
      get_uuid_str(uuid_str, &g_devices_info[i].uuid);
      cJSON *limit = cJSON_GetObjectItem(gpu_limits, uuid_str);
      if (limit != NULL)
      {
        gpu_mem_limit[i] = limit->valueint;
        gpu_mem_limit[i] = gpu_mem_limit[i] * 1024 * 1024;
      }
    }
    g_anycuda_config.gpu_mem_limit_valid = 1;
    for (int i = 0; i < g_device_count; i++)
    {
      g_anycuda_config.gpu_mem_limit[i] = gpu_mem_limit[i];
    }
  }

  LOGGER(VERBOSE, "pod name         : %s", g_anycuda_config.pod_name);
  LOGGER(VERBOSE, "resource name    : %s", g_anycuda_config.resource_name);
  LOGGER(VERBOSE, "gpu count        : %d", g_anycuda_config.gpu_count);
  for (int i = 0; i < g_anycuda_config.gpu_count; i++)
  {
    LOGGER(VERBOSE, "gpu-%d-%s: %zu", i, g_anycuda_config.gpu_uuids[i], g_anycuda_config.gpu_mem_limit[i]);
  }
  g_anycuda_config.valid = 1;

  ret = 0;
DONE:
  if (likely(fd))
  {
    close(fd);
  }

  return ret;
}

int split_str(char *line, char *key, char *value, char d)
{
  int index = 0;
  for (index = 0; index < strlen(line) && line[index] != d; index++)
  {
  }

  if (index == strlen(line))
  {
    key[0] = '\0';
    value = '\0';
    return 1;
  }

  int start = 0, i = 0;
  // trim head
  for (; start < index && (line[start] == ' ' || line[start] == '\t'); start++)
  {
  }

  for (i = 0; start < index; i++, start++)
  {
    key[i] = line[start];
  }
  // trim tail
  for (; i > 0 && (key[i - 1] == '\0' || key[i - 1] == '\n' || key[i - 1] == '\t'); i--)
  {
  }
  key[i] = '\0';

  start = index + 1;
  i = 0;

  // trim head
  for (; start < strlen(line) && (line[start] == ' ' || line[start] == '\t'); start++)
  {
  }

  for (i = 0; start < strlen(line); i++, start++)
  {
    value[i] = line[start];
  }
  // trim tail
  for (; i > 0 && (value[i - 1] == '\0' || value[i - 1] == '\n' || value[i - 1] == '\t'); i--)
  {
  }
  value[i] = '\0';
  return 0;
}

int read_cgroup(char *pidpath, char *cgroup_key, char *cgroup_value)
{
  char buff[255];
  FILE *f = fopen(pidpath, "rb");
  if (f == NULL)
  {
    LOGGER(VERBOSE, "read file %s failed\n", pidpath);
    return 1;
  }

  while (fgets(buff, 255, f))
  {
    int index = 0;
    for (; index < strlen(buff) && buff[index] != ':'; index++)
    {
    }

    if (index == strlen(buff))
      continue;

    char key[128], value[128];
    if (split_str(&buff[index + 1], key, value, ':') != 0)
      continue;

    if (strcmp(key, cgroup_key) == 0)
    {
      strcpy(cgroup_value, value);
      fclose(f);
      return 0;
    }
  }

  fclose(f);
  return 1;
}

int check_in_pod()
{
  DIR *proc_dir = opendir("/host_proc/");
  if (proc_dir)
  {
    closedir(proc_dir);
    return 0;
  }
  return 1;
}

int check_pod_pid(unsigned int pid)
{
  if (pid == 0)
    return 1;

  char pidpath[128] = "";
  sprintf(pidpath, "/host_proc/%d/cgroup", pid);

  char pod_cg[256];
  char process_cg[256];

  if ((read_cgroup("/proc/1/cgroup", "memory", pod_cg) == 0) && (read_cgroup(pidpath, "memory", process_cg) == 0))
  {
    LOGGER(VERBOSE, "pod cg: %s\nprocess_cg: %s\n", pod_cg, process_cg);
    if (strstr(process_cg, pod_cg) != NULL)
    {
      LOGGER(VERBOSE, "cg match");
      return 0;
    }
  }
  LOGGER(VERBOSE, "cg mismatch");
  return 1;
}

static void get_used_gpu_memory(void *arg, CUdevice device_id)
{
  size_t *used_memory = arg;

  nvmlDevice_t dev;
  nvmlProcessInfo_t pids_on_device[MAX_PIDS];
  unsigned int size_on_device = MAX_PIDS;
  int ret;

  unsigned int i;

  ret =
      NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetHandleByIndex, device_id, &dev);
  if (unlikely(ret))
  {
    LOGGER(FATAL, "nvmlDeviceGetHandleByIndex can't find device %d, return %d", device_id, ret);
    *used_memory = 0;
    return;
  }

  ret =
      NVML_ENTRY_CALL(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses,
                      dev, &size_on_device, pids_on_device);
  if (unlikely(ret))
  {
    LOGGER(FATAL,
           "nvmlDeviceGetComputeRunningProcesses can't get pids on device 0, "
           "return %d",
           ret);
    *used_memory = 0;
    return;
  }

  if (check_in_pod() == 0)
  {
    for (i = 0; i < size_on_device; i++)
    {
      if (check_pod_pid(pids_on_device[i].pid) == 0)
      {
        LOGGER(VERBOSE, "pid[%d] use memory: %lld", pids_on_device[i].pid,
               pids_on_device[i].usedGpuMemory);
        *used_memory += pids_on_device[i].usedGpuMemory;
      }
    }
  }
  else
  {
    for (i = 0; i < size_on_device; i++)
    {
      LOGGER(VERBOSE, "pid[%d] use memory: %lld", pids_on_device[i].pid,
             pids_on_device[i].usedGpuMemory);
      *used_memory += pids_on_device[i].usedGpuMemory;
    }
  }

  LOGGER(VERBOSE, "total used memory: %zu", *used_memory);
}

void get_uuid_str(char *dest, CUuuid *src)
{
  size_t n = 0, i = 0;
  dest[0] = 'G';
  dest[1] = 'P';
  dest[2] = 'U';
  dest[3] = '-';
  n = 4;
  while (i < 4)
  {
    sprintf(dest + n, "%02x", (unsigned char)src->bytes[i++]);
    n += 2;
  }
  dest[n++] = '-';
  while (i < 6)
  {
    sprintf(dest + n, "%02x", (unsigned char)src->bytes[i++]);
    n += 2;
  }
  dest[n++] = '-';
  while (i < 8)
  {
    sprintf(dest + n, "%02x", (unsigned char)src->bytes[i++]);
    n += 2;
  }
  dest[n++] = '-';
  while (i < 10)
  {
    sprintf(dest + n, "%02x", (unsigned char)src->bytes[i++]);
    n += 2;
  }
  dest[n++] = '-';
  while (i < 16)
  {
    sprintf(dest + n, "%02x", (unsigned char)src->bytes[i++]);
    n += 2;
  }
}

static void load_devices_info()
{
  CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetCount, &g_device_count);
  for (int i = 0; i < g_device_count; i++)
  {
    CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGet, &g_devices_info[i].device, i);
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuDeviceGetUuid_v2) != NULL)
    {
      CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetUuid_v2, &g_devices_info[i].uuid, g_devices_info[i].device);
    }
    else
    {
      CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetUuid, &g_devices_info[i].uuid, g_devices_info[i].device);
    }
  }
}

static void initialization()
{
  int ret;
  const char *cuda_err_string = NULL;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, 0);
  if (unlikely(ret))
  {
    LOGGER(FATAL, "cuInit error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }

  load_devices_info();
  read_anylearn_podconf();
  active_podconf_notifier();
}

/** hijack entrypoint */
CUresult cuDriverGetVersion(int *driverVersion)
{
  CUresult ret;

  load_necessary_data();
  pthread_once(&g_init_set, initialization);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDriverGetVersion, driverVersion);
  if (unlikely(ret))
  {
    goto DONE;
  }

DONE:
  return ret;
}

CUresult cuInit(unsigned int flag)
{
  CUresult ret;

  load_necessary_data();
  pthread_once(&g_init_set, initialization);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, flag);

  if (unlikely(ret))
  {
    goto DONE;
  }

DONE:
  return ret;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags)
{
  size_t used = 0;
  size_t request_size = bytesize;
  CUresult ret;

  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    CUdevice ordinal;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &ordinal);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    get_used_gpu_memory((void *)&used, ordinal);

    if (g_anycuda_config.gpu_mem_limit[ordinal] >= 0 && used + request_size > g_anycuda_config.gpu_mem_limit[ordinal])
    {
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        flags);
DONE:
  return ret;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
  size_t used = 0;
  size_t request_size = bytesize;
  CUresult ret;

  if (g_anycuda_config.valid)
  {
    if (!g_anycuda_config.gpu_mem_limit_valid)
    {
      LOGGER(VERBOSE, "gpuLimit is not valid now, use host memory");
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
    CUdevice ordinal;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &ordinal);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    get_used_gpu_memory((void *)&used, ordinal);

    if (g_anycuda_config.gpu_mem_limit[ordinal] >= 0 && (used + request_size > g_anycuda_config.gpu_mem_limit[ordinal]))
    {
      LOGGER(WARNING, "has used more gpu mem than limit on device %d: %lu >= %lu", ordinal, used + request_size, g_anycuda_config.gpu_mem_limit[ordinal]);
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAlloc_v2, dptr, bytesize);
DONE:
  return ret;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
  size_t used = 0;
  size_t request_size = bytesize;
  CUresult ret;

  if (g_anycuda_config.valid)
  {
    if (!g_anycuda_config.gpu_mem_limit_valid)
    {
      LOGGER(VERBOSE, "gpuLimit is not valid now, use host memory");
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
    CUdevice ordinal;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &ordinal);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    get_used_gpu_memory((void *)&used, ordinal);

    if (g_anycuda_config.gpu_mem_limit[ordinal] >= 0 && used + request_size > g_anycuda_config.gpu_mem_limit[ordinal])
    {
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAlloc, dptr, bytesize);
DONE:
  return ret;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes)
{
  size_t used = 0;
  size_t request_size = ROUND_UP(WidthInBytes * Height, ElementSizeBytes);
  CUresult ret;

  if (g_anycuda_config.valid)
  {
    if (!g_anycuda_config.gpu_mem_limit_valid)
    {
      LOGGER(VERBOSE, "gpuLimit is not valid now, use host memory");
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, request_size, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
    CUdevice ordinal;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &ordinal);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    get_used_gpu_memory((void *)&used, ordinal);

    if (g_anycuda_config.gpu_mem_limit[ordinal] >= 0 && used + request_size > g_anycuda_config.gpu_mem_limit[ordinal])
    {
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, request_size, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocPitch_v2, dptr, pPitch,
                        WidthInBytes, Height, ElementSizeBytes);
DONE:
  return ret;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes)
{
  size_t used = 0;
  size_t request_size = ROUND_UP(WidthInBytes * Height, ElementSizeBytes);
  CUresult ret;

  if (g_anycuda_config.valid)
  {
    if (!g_anycuda_config.gpu_mem_limit_valid)
    {
      LOGGER(VERBOSE, "gpuLimit is not valid now, use host memory");
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, request_size, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
    CUdevice ordinal;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &ordinal);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    get_used_gpu_memory((void *)&used, ordinal);

    if (g_anycuda_config.gpu_mem_limit[ordinal] >= 0 && used + request_size > g_anycuda_config.gpu_mem_limit[ordinal])
    {
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, request_size, CU_MEM_ATTACH_GLOBAL);
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocPitch, dptr, pPitch,
                        WidthInBytes, Height, ElementSizeBytes);
DONE:
  return ret;
}

static size_t get_array_base_size(int format)
{
  size_t base_size = 0;

  switch (format)
  {
  case CU_AD_FORMAT_UNSIGNED_INT8:
  case CU_AD_FORMAT_SIGNED_INT8:
    base_size = 8;
    break;
  case CU_AD_FORMAT_UNSIGNED_INT16:
  case CU_AD_FORMAT_SIGNED_INT16:
  case CU_AD_FORMAT_HALF:
    base_size = 16;
    break;
  case CU_AD_FORMAT_UNSIGNED_INT32:
  case CU_AD_FORMAT_SIGNED_INT32:
  case CU_AD_FORMAT_FLOAT:
    base_size = 32;
    break;
  default:
    base_size = 32;
  }

  return base_size;
}

static CUresult
cuArrayCreate_helper(const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
  size_t used = 0;
  size_t base_size = 0;
  size_t request_size = 0;
  CUresult ret = CUDA_SUCCESS;

  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    CUdevice device_id;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device_id);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    base_size = get_array_base_size(pAllocateArray->Format);
    request_size = base_size * pAllocateArray->NumChannels *
                   pAllocateArray->Height * pAllocateArray->Width;

    get_used_gpu_memory((void *)&used, device_id);

    if (g_anycuda_config.gpu_mem_limit[device_id] >= 0 && used + request_size > g_anycuda_config.gpu_mem_limit[device_id])
    {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

DONE:
  return ret;
}

CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
  CUresult ret;

  ret = cuArrayCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS)
  {
    goto DONE;
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate_v2, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
  CUresult ret;

  ret = cuArrayCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS)
  {
    goto DONE;
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

static CUresult
cuArray3DCreate_helper(const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
  size_t used = 0;
  size_t base_size = 0;
  size_t request_size = 0;
  CUresult ret = CUDA_SUCCESS;

  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    CUdevice device_id;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device_id);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    base_size = get_array_base_size(pAllocateArray->Format);
    request_size = base_size * pAllocateArray->NumChannels *
                   pAllocateArray->Height * pAllocateArray->Width;

    get_used_gpu_memory((void *)&used, device_id);

    if (g_anycuda_config.gpu_mem_limit[device_id] >= 0 && used + request_size > g_anycuda_config.gpu_mem_limit[device_id])
    {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

DONE:
  return ret;
}

CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
  CUresult ret;

  ret = cuArray3DCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS)
  {
    goto DONE;
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate_v2, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
  CUresult ret;

  ret = cuArray3DCreate_helper(pAllocateArray);
  if (ret != CUDA_SUCCESS)
  {
    goto DONE;
  }
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate, pHandle,
                        pAllocateArray);
DONE:
  return ret;
}

CUresult
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels)
{
  size_t used = 0;
  size_t base_size = 0;
  size_t request_size = 0;
  CUresult ret;

  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    CUdevice device_id;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device_id);
    if (ret != CUDA_SUCCESS)
    {
      goto DONE;
    }
    base_size = get_array_base_size(pMipmappedArrayDesc->Format);
    request_size = base_size * pMipmappedArrayDesc->NumChannels *
                   pMipmappedArrayDesc->Height * pMipmappedArrayDesc->Width *
                   pMipmappedArrayDesc->Depth;

    get_used_gpu_memory((void *)&used, device_id);

    if (g_anycuda_config.gpu_mem_limit[device_id] >= 0 && used + request_size > g_anycuda_config.gpu_mem_limit[device_id])
    {
      ret = CUDA_ERROR_OUT_OF_MEMORY;
      goto DONE;
    }
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMipmappedArrayCreate, pHandle,
                        pMipmappedArrayDesc, numMipmapLevels);
DONE:
  return ret;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    *bytes = g_anycuda_config.gpu_mem_limit[dev];

    return CUDA_SUCCESS;
  }

  return CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    *bytes = g_anycuda_config.gpu_mem_limit[dev];

    return CUDA_SUCCESS;
  }

  return CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem, bytes, dev);
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total)
{
  size_t used = 0;
  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    CUdevice device_id;
    CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device_id);
    if (ret != CUDA_SUCCESS)
    {
      return ret;
    }
    get_used_gpu_memory((void *)&used, device_id);

    *total = g_anycuda_config.gpu_mem_limit[device_id];
    *free =
        used > g_anycuda_config.gpu_mem_limit[device_id] ? 0 : g_anycuda_config.gpu_mem_limit[device_id] - used;

    return CUDA_SUCCESS;
  }

  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo_v2, free, total);
}

CUresult cuMemGetInfo(size_t *free, size_t *total)
{
  size_t used = 0;

  if (g_anycuda_config.valid && g_anycuda_config.gpu_mem_limit_valid)
  {
    CUdevice device_id;
    CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device_id);
    if (ret != CUDA_SUCCESS)
    {
      return ret;
    }
    get_used_gpu_memory((void *)&used, device_id);

    *total = g_anycuda_config.gpu_mem_limit[device_id];
    *free =
        used > g_anycuda_config.gpu_mem_limit[device_id] ? 0 : g_anycuda_config.gpu_mem_limit[device_id] - used;

    return CUDA_SUCCESS;
  }

  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo, free, total);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel_ptsz, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunch(CUfunction f)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunch, f);
}

CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel_ptsz, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void **kernelParams)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGrid, f, grid_width,
                         grid_height);
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGridAsync, f, grid_width,
                         grid_height, hStream);
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
{
  return CUDA_ENTRY_CALL(cuda_library_entry, cuFuncSetBlockShape, hfunc, x, y,
                         z);
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags)
{
  CUresult ret;
  int i;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn,
                        cudaVersion, flags);
  if (ret == CUDA_SUCCESS)
  {
    for (i = 0; i < cuda_hook_nums; i++)
    {
      if (!strcmp(symbol, cuda_hooks_entry[i].name))
      {
        LOGGER(5, "Match hook %s", symbol);
        *pfn = cuda_hooks_entry[i].fn_ptr;
        break;
      }
    }
  }

  return ret;
}
