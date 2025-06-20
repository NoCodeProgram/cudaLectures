#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "stb_image_write.h"

__global__ void invertKernel(uint8_t* imgPtr)
{
    int32_t x = threadIdx.x;
    int32_t y = threadIdx.y;
    int32_t idx = y * 32 + x;

    imgPtr[idx] = 255 - imgPtr[idx];
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
    
    uint8_t* deviceImgPtr;
    cudaMalloc(&deviceImgPtr, imgBytes);

    cudaMemcpy(deviceImgPtr, hostImage, imgBytes, cudaMemcpyHostToDevice);

    constexpr dim3 blockSize(32, 32);
    invertKernel<<<1, blockSize>>>(deviceImgPtr);
    cudaDeviceSynchronize();
    
    cudaMemcpy(hostImage, deviceImgPtr, imgBytes, cudaMemcpyDeviceToHost);

    stbi_write_png("inverted_cat32gray.png", imgWidth, imgHeight, imgChannels, hostImage, imgWidth * sizeof(uint8_t));

    cudaFree(deviceImgPtr);
    stbi_image_free(hostImage);
    
    std::cout << "Image inversion completed!" << std::endl;
    return 0;
}