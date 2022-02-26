#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

#define DATA_SIZE 1048576

int src[DATA_SIZE];
int dest[DATA_SIZE];

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
    CUdevice cuDevice = 0;
    cuDeviceGet(&cuDevice, 0);

    //创建上下文
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);

    //显存中分配向量空间
    size_t size = DATA_SIZE * sizeof(float);
    CUdeviceptr d_A;
    cuMemAlloc_v2(&d_A, size);

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