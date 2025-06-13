#include <iostream>
#include <chrono>
#include <random>
#include <numeric>
#include <cuda_runtime.h>

// CUDA kernel to transform elements
__global__ void multiply10(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 10.0f;
    }
}

int main()
{
    constexpr uint64_t numElements = 2'000'000'000;
    constexpr size_t bytes = numElements * sizeof(float);

    std::cout << "Allocating " << numElements << " float elements (~8GB)..." << std::endl;

    constexpr uint64_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);
   
    float *hData = new float[numElements];
    for (uint64_t i = 0; i < numElements; i++)
    {
        hData[i] = dist(gen);
    }
    
    float *dData = nullptr;
    cudaMalloc(&dData, bytes);
    cudaMemcpy(dData, hData, bytes, cudaMemcpyHostToDevice);
    
    // Set up CUDA kernel execution
    const int blockSize = 256;
    const int gridSize = (numElements + blockSize - 1) / blockSize;
    
    const auto start = std::chrono::high_resolution_clock::now();
    
    multiply10<<<gridSize, blockSize>>>(dData, numElements);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = end - start;
    
    // Copy result back to host
    cudaMemcpy(hData, dData, bytes, cudaMemcpyDeviceToHost);

    std::cout << "First and last element: " << hData[0] << " " << hData[numElements - 1] << std::endl;
    std::cout << "Computation completed in " << totalTime.count() << " seconds" << std::endl;
    
    // Clean up
    cudaFree(dData);
    delete[] hData;
    
    return 0;
}