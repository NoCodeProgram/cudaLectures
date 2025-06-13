#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void add100(int32_t* data)
{
    const int idx = threadIdx.x;
    data[idx] = data[idx] + 100;
}

void cpuAdd100(int32_t * data)
{
    for(int32_t idx = 0 ; idx < 128 ; ++idx)
    {
        data[idx] = data[idx] + 100;
    }
}

int main()
{
    constexpr uint32_t dataLength = 128;
    int32_t *hostData = new int32_t[dataLength];
    for (int32_t i = 0; i < dataLength; ++i)
    {
        hostData[i] = i; 
    }

    int32_t* deviceData;
    cudaMalloc(&deviceData, dataLength * sizeof(int32_t));
    
    cudaMemcpy(deviceData, hostData, dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    add100 <<<1, dataLength >>> (deviceData);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(hostData, deviceData, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(deviceData);

    for (int32_t i = 0; i < dataLength ; ++i)
    {
        std::cout << hostData[i] << " ";
    }
    
    delete [] hostData;

    return 0;
}