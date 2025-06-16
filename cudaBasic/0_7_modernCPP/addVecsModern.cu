
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int32_t* dataA, const int32_t* dataB, int32_t* dataC)
{
    const int idx = threadIdx.x;
    dataC[idx] = dataA[idx] + dataB[idx];
}

int main()
{
    constexpr uint32_t dataLength = 1024;

    std::vector<int32_t> hostDataA(dataLength);
    std::vector<int32_t> hostDataB(dataLength);
    std::vector<int32_t> hostDataC(dataLength);

// Initialize data
    for (int32_t i = 0; i < dataLength; ++i)
    {
        hostDataA[i] = i;// A = [0, 1, 2, 3, ...]
        hostDataB[i] = i * 2;// B = [0, 2, 4, 6, ...]
        hostDataC[i] = 0;// C = [0, 0, 0, 0, ...]
    }

// Allocate device memory
    int32_t* deviceDataA = nullptr;
    int32_t* deviceDataB = nullptr;
    int32_t* deviceDataC = nullptr;

    cudaMalloc(&deviceDataA, dataLength * sizeof(int32_t));
    cudaMalloc(&deviceDataB, dataLength * sizeof(int32_t));
    cudaMalloc(&deviceDataC, dataLength * sizeof(int32_t));

// Copy host to device memory
    cudaMemcpy(deviceDataA, hostDataA.data(), dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDataB, hostDataB.data(), dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);

// Launch kernel
    vectorAdd <<<1, dataLength >>> (deviceDataA, deviceDataB, deviceDataC);

// Synchronize
    cudaDeviceSynchronize();

// Copy device to host memory
    cudaMemcpy(hostDataC.data(), deviceDataC, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost);

// Print results (first 10 and last 10 elements)
    std::cout << "First 10 : ";
    for (int32_t i = 0; i < 10; ++i)
    {
        std::cout << hostDataC[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Last 10 : ";
    for (int32_t i = dataLength - 10; i < static_cast<int32_t>(dataLength); ++i)
    {
        std::cout << hostDataC[i] << " ";
    }
    std::cout << std::endl;

// Free memory
    cudaFree(deviceDataA);
    cudaFree(deviceDataB);
    cudaFree(deviceDataC);

    return 0;
}
