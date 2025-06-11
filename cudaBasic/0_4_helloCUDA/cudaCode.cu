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
    constexpr uint64_t num_elements = 2'000'000'000;
    const size_t bytes = num_elements * sizeof(float);

    std::cout << "Allocating " << num_elements << " float elements (~8GB)..." << std::endl;

    constexpr uint64_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);
   
    // Host data
    float *h_data = new float[num_elements];
    for (uint64_t i = 0; i < num_elements; i++)
    {
        h_data[i] = dist(gen);
    }
    
    // Device data
    float *d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    // Set up CUDA kernel execution
    const int blockSize = 256;
    const int gridSize = (num_elements + blockSize - 1) / blockSize;
    
    const auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    multiply10<<<gridSize, blockSize>>>(d_data, num_elements);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;
    
    // Copy result back to host
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    std::cout << "First and last element: " << h_data[0] << " " << h_data[num_elements - 1] << std::endl;
    std::cout << "Computation completed in " << total_time.count() << " seconds" << std::endl;
    
    // Clean up
    cudaFree(d_data);
    delete[] h_data;
    
    return 0;
}