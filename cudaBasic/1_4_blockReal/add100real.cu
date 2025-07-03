#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void add100(int32_t* data, uint32_t dataLength) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataLength) {
        data[idx] = data[idx] + 100;
    }
}

int main()
{
    uint32_t dataLength;
    
    std::cout << "Enter the data length: ";
    std::cin >> dataLength;
    
    if (dataLength == 0) {
        std::cerr << "Error: Data length must be greater than 0" << std::endl;
        return 1;
    }
    
    std::vector<int32_t> hostData(dataLength);
    for (uint32_t i = 0; i < dataLength; ++i)
    {
        hostData[i] = static_cast<int32_t>(i);
    }

    int32_t* deviceData;
    cudaMalloc(&deviceData, dataLength * sizeof(int32_t));    
    cudaMemcpy(deviceData, hostData.data(), dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    constexpr int32_t blockSize = 1024;
    const int32_t numBlocks = (dataLength + blockSize - 1) / blockSize;
    
    std::cout << "Processing " << dataLength << " elements with " << numBlocks << " blocks of " << blockSize << " threads each" << std::endl;
    
    add100<<<numBlocks, blockSize>>>(deviceData, dataLength);

    cudaDeviceSynchronize();
    cudaMemcpy(hostData.data(), deviceData, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceData);

    // Print last 10 elements (or all if less than 10)
    const uint32_t startIdx = std::max(0u, dataLength - 10);
    std::cout << "Last " << (dataLength - startIdx) << " elements: ";
    for(uint32_t idx = startIdx; idx < dataLength; ++idx)
    {
        std::cout << hostData[idx] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}