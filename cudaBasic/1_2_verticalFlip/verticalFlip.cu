#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "stb_image_write.h"

__global__ void verticalFlipKernel(const uint8_t* input, uint8_t* output)
{
    int32_t x = threadIdx.x;  // 0 ~ 31
    int32_t y = threadIdx.y;  // 0 ~ 31
    
    // Calculate original pixel position
    int32_t inputIdx = y * 32 + x;
    
    // Calculate vertically flipped pixel position
    int32_t flippedY = 31 - y;  // y=0 → 31, y=1 → 30, ..., y=31 → 0
    int32_t outputIdx = flippedY * 32 + x;
    
    // Copy pixel (only position changes)
    output[outputIdx] = input[inputIdx];
}

int main()
{
    // Load image
    int imgWidth, imgHeight, imgChannels;
    uint8_t* hostImage = stbi_load("cat32gray.png", 
                                   &imgWidth, &imgHeight, &imgChannels, 1);

    assert(imgWidth == 32 && imgHeight == 32 && imgChannels == 1);
    
    constexpr int32_t imgSize = 32 * 32;
    constexpr size_t imgBytes = imgSize * sizeof(uint8_t);
    
    // Allocate GPU memory (for input and output)
    uint8_t* deviceInputPtr;
    uint8_t* deviceOutputPtr;
    cudaMalloc(&deviceInputPtr, imgBytes);
    cudaMalloc(&deviceOutputPtr, imgBytes);

    // Copy original image to GPU
    cudaMemcpy(deviceInputPtr, hostImage, imgBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    constexpr dim3 blockSize(32, 32);
    verticalFlipKernel<<<1, blockSize>>>(deviceInputPtr, deviceOutputPtr);
    cudaDeviceSynchronize();
    
    // Copy result back to CPU
    cudaMemcpy(hostImage, deviceOutputPtr, imgBytes, cudaMemcpyDeviceToHost);

    // Save vertically flipped image
    stbi_write_png("flipped_cat32gray.png", imgWidth, imgHeight, 
                   imgChannels, hostImage, imgWidth * sizeof(uint8_t));

    // Free memory
    cudaFree(deviceInputPtr);
    cudaFree(deviceOutputPtr);
    stbi_image_free(hostImage);
    
    std::cout << "Vertical flip completed!" << std::endl;
    return 0;
}