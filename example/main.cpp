#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

#define DATA_SIZE 104748364

int src[DATA_SIZE];
int dest[DATA_SIZE];

CUdeviceptr onDevicePtr[60];

int main()
{
    //初始化设备
    if (cuInit(0) != CUDA_SUCCESS)
        exit(0);

    //获得支持cuda的设备数目
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0)
        exit(0);

    //获得设备０句柄
    CUdevice cuDevice0, cuDevice1;
    cuDeviceGet(&cuDevice0, 0);
    cuDeviceGet(&cuDevice1, 1);

    //创建上下文
    CUcontext cuContext0, cuContext1;
    cuCtxCreate_v2(&cuContext0, 0, cuDevice0);
    cuCtxCreate_v2(&cuContext1, 0, cuDevice1);

    //显存中分配向量空间
    size_t size = DATA_SIZE * sizeof(float);
    CUdeviceptr d_A, d_B;
    cuCtxSetCurrent(cuContext0);
    cuMemAlloc_v2(&d_A, size);
    for (int i = 0; i < 10; i++)
    {
        cuMemAlloc_v2(&onDevicePtr[i], size);
    }
    for (int i = 10; i < 20; i++)
    {
        cuMemAlloc_v2(&onDevicePtr[i], size);
    }
    for (int i = 20; i < 30; i++)
    {
        cuMemAlloc_v2(&onDevicePtr[i], size);
    }
    for (int i = 30; i < 40; i++)
    {
        cuMemAlloc_v2(&onDevicePtr[i], size);
    }
    for (int i = 40; i < 50; i++)
    {
        cuMemAlloc_v2(&onDevicePtr[i], size);
    }

    cuCtxSetCurrent(cuContext1);
    cuMemAlloc_v2(&d_B, size);

    //初始化host的变量
    for (int i = 0; i < DATA_SIZE; i++)
    {
        src[i] = i;
    }

    //从内存向显存拷贝向量
    cuMemcpyHtoD_v2(d_A, src, size);

    //从显存向内存拷回结果
    cuMemcpyDtoH_v2(dest, d_A, size);

    int flag = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        flag += dest[i] - src[i];
    }
    if (flag != 0)
    {
        printf("all elements are equal");
    }
    else
    {
        printf("some elements are not equal");
    }

    sleep(100);

    //释放显存空间
    cuMemFree(d_A);
    return 0;
}