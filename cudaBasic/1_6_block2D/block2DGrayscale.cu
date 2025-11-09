#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <vector>

__global__ void colorToGrayscaleKernel(const uint8_t* colorInput, uint8_t* grayOutput, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int colorIdx = (y * width + x) * 3;
    const int grayIdx = y * width + x;

    const float r = static_cast<float>(colorInput[colorIdx + 0]);
    const float g = static_cast<float>(colorInput[colorIdx + 1]);
    const float b = static_cast<float>(colorInput[colorIdx + 2]);

    const float gray = 0.299f * r + 0.587f * g + 0.114f * b;

    grayOutput[grayIdx] = static_cast<uint8_t>(gray);
}

int main()
{
    // Load 1000x1000 color image (3 channels)
    int imgWidth, imgHeight, imgChannels;
    uint8_t* hostColorImage = stbi_load("cat1000color.png",
        &imgWidth, &imgHeight, &imgChannels, 3);

    if (!hostColorImage) {
        std::cerr << "Error: Failed to load cat1000color.png" << std::endl;
        return 1;
    }

    assert(imgChannels == 3);

    std::cout << "Loaded image: " << imgWidth << "x" << imgHeight
              << " with " << imgChannels << " channels" << std::endl;

    // Allocate GPU memory
    uint8_t* deviceColorInput;
    uint8_t* deviceGrayOutput;
    const size_t colorImageSize = imgWidth * imgHeight * 3 * sizeof(uint8_t);
    const size_t grayImageSize = imgWidth * imgHeight * sizeof(uint8_t);

    cudaMalloc(&deviceColorInput, colorImageSize);
    cudaMalloc(&deviceGrayOutput, grayImageSize);

    // Copy color image to GPU
    cudaMemcpy(deviceColorInput, hostColorImage, colorImageSize, cudaMemcpyHostToDevice);

    // Configure 2D grid and block dimensions
    constexpr int BLOCK_SIZE = 32;
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);  // 32x32 threads per block
    const dim3 gridSize(
        (imgWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,   // 1000/32 = 32 blocks in x
        (imgHeight + BLOCK_SIZE - 1) / BLOCK_SIZE   // 1000/32 = 32 blocks in y
    );

    std::cout << "Launching kernel with grid (" << gridSize.x << ", " << gridSize.y
              << ") and block (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;

    colorToGrayscaleKernel<<<gridSize, blockSize>>>(deviceColorInput, deviceGrayOutput, imgWidth, imgHeight);
    cudaDeviceSynchronize();

    // Copy grayscale result back to CPU
    std::vector<uint8_t> hostGrayResult(imgWidth * imgHeight);
    cudaMemcpy(hostGrayResult.data(), deviceGrayOutput, grayImageSize, cudaMemcpyDeviceToHost);

    // 흑백 이미지 저장
    stbi_write_png("cat1000gray_converted.png", imgWidth, imgHeight, 1, hostGrayResult.data(), imgWidth);

    // 메모리 해제
    cudaFree(deviceColorInput);
    cudaFree(deviceGrayOutput);
    stbi_image_free(hostColorImage);

    std::cout << "Color to grayscale conversion completed!" << std::endl;
    std::cout << "Output saved to: cat1000gray_converted.png" << std::endl;
    return 0;
}
