#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <source_location>

inline void cudaCheckErr(cudaError_t err, const std::source_location& loc = std::source_location::current())
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << loc.file_name() << ":" << loc.line() 
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void add100(int32_t* data)
{
    const int idx = threadIdx.x;
    data[idx] = data[idx] + 100;
}

int main()
{
    constexpr uint32_t dataLength = 1025;
    std::vector<int32_t> hostData(dataLength);
    for (int32_t i = 0; i < dataLength; ++i)
    {
        hostData[i] = i; 
    }

    int32_t* deviceData;
    const auto mallocErr = cudaMalloc(&deviceData, dataLength * sizeof(int32_t));
    cudaCheckErr(mallocErr);

    cudaCheckErr(cudaMemcpy(deviceData, hostData.data(), dataLength * sizeof(int32_t), cudaMemcpyHostToDevice));
    add100 <<<1, dataLength >>> (deviceData);
    const cudaError_t launchErr = cudaGetLastError();    
    cudaCheckErr(launchErr);

    cudaCheckErr(cudaDeviceSynchronize());

    cudaCheckErr(cudaMemcpy(hostData.data(), deviceData, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(deviceData);

    for (int32_t i = 0; i < 10; ++i)
    {
        std::cout << hostData[i] << " ";
    }    
    return 0;
}