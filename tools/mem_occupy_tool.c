#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>

int str2int(char *str) {
    int l = strlen(str);
    int res = 0;
    for (int i = 0; i < l; i++) {
        res *= 10;
        res += str[i] - '0';
    }
    return res;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("wrong arguments: mem_occupy_tool device_index size(MB)");
        return -1;
    }
    int device_id = str2int(argv[1]);
    int size_mb = str2int(argv[2]);

    //初始化设备
    if (cuInit(0) != CUDA_SUCCESS) return -1;

    //获得设备句柄
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, device_id);

    //创建上下文
    CUcontext cuContext;
    cuCtxCreate_v2(&cuContext, 0, cuDevice);

    //显存中分配向量空间
    size_t size = size_mb;
    size *= 1024 * 1024;
    CUdeviceptr d;
    cuCtxSetCurrent(cuContext);
    cuMemAlloc_v2(&d, size);
    while (1)
    {
    }
    cuMemFree_v2(d);
    return 0;
}