#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void add100(int32_t* data) 
{
    const int idx = threadIdx.x;
    data[idx] = data[idx] + 100;
}

int main()
{
    constexpr uint32_t dataLength = 1024 * 100;
    std::vector<int32_t> hostData(dataLength);
    
    for (uint32_t i = 0; i < dataLength; ++i)
    {
        hostData[i] = static_cast<int32_t>(i);
    }

    int32_t* deviceData;
    cudaMalloc(&deviceData, dataLength * sizeof(int32_t));    
    cudaMemcpy(deviceData, hostData.data(), dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    constexpr int32_t blockSize = 1024;
    constexpr int32_t numBlocks = 100;
    add100<<<numBlocks, blockSize>>>(deviceData);

    cudaDeviceSynchronize();
    cudaMemcpy(hostData.data(), deviceData, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceData);

    for(uint32_t idx = datalength - 10; idx < dataLength; ++idx)
    {
        std::cout << hostData[idx] << " ";
    }
    std::cout << std::endl;
    return 0;
}