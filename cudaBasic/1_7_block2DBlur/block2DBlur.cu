#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <vector>

__global__ void blur5x5Kernel(const uint8_t* input, uint8_t* output, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // 5x5 average blur
    float sum = 0.0f;
    int count = 0;

    // Iterate over 5x5 neighborhood
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            const int nx = x + dx;
            const int ny = y + dy;

            // Skip if out of bounds
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            const int idx = ny * width + nx;
            sum += static_cast<float>(input[idx]);
            count++;
        }
    }

    // Calculate average
    const int outIdx = y * width + x;
    output[outIdx] = static_cast<uint8_t>(sum / count);
}

int main()
{
    // Load 1000x1000 grayscale image (1 channel)
    int imgWidth, imgHeight, imgChannels;
    uint8_t* hostGrayImage = stbi_load("cat1000gray.png",
        &imgWidth, &imgHeight, &imgChannels, 1);

    if (!hostGrayImage) {
        std::cerr << "Error: Failed to load cat1000gray.png" << std::endl;
        return 1;
    }

    assert(imgChannels == 1);

    std::cout << "Loaded image: " << imgWidth << "x" << imgHeight
              << " with " << imgChannels << " channel" << std::endl;

    // Allocate GPU memory
    uint8_t* deviceInput;
    uint8_t* deviceOutput;
    const size_t imageSize = imgWidth * imgHeight * sizeof(uint8_t);

    cudaMalloc(&deviceInput, imageSize);
    cudaMalloc(&deviceOutput, imageSize);

    // Copy grayscale image to GPU
    cudaMemcpy(deviceInput, hostGrayImage, imageSize, cudaMemcpyHostToDevice);

    // Configure 2D grid and block dimensions
    constexpr int BLOCK_SIZE = 32;
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);  // 32x32 threads per block
    const dim3 gridSize(
        (imgWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,   // 1000/32 = 32 blocks in x
        (imgHeight + BLOCK_SIZE - 1) / BLOCK_SIZE   // 1000/32 = 32 blocks in y
    );

    std::cout << "Launching kernel with grid (" << gridSize.x << ", " << gridSize.y
              << ") and block (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;

    blur3x3Kernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, imgWidth, imgHeight);
    cudaDeviceSynchronize();

    // Copy blurred result back to CPU
    std::vector<uint8_t> hostBlurredResult(imgWidth * imgHeight);
    cudaMemcpy(hostBlurredResult.data(), deviceOutput, imageSize, cudaMemcpyDeviceToHost);

    // Save blurred image
    stbi_write_png("cat1000gray_blurred.png", imgWidth, imgHeight, 1, hostBlurredResult.data(), imgWidth);

    // Free memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    stbi_image_free(hostGrayImage);

    std::cout << "3x3 average blur completed!" << std::endl;
    std::cout << "Output saved to: cat1000gray_blurred.png" << std::endl;
    return 0;
}
