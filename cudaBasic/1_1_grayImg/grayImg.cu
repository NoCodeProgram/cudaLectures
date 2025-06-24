#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "stb_image_write.h"
#include <iostream>
#include <vector>

__global__ void colorToGrayscaleKernel(const uint8_t* colorInput, uint8_t* grayOutput)
{
    const int x = threadIdx.x;  // 0 ~ 31
    const int y = threadIdx.y;  // 0 ~ 31
    
    const int colorIdx = (y * 32 + x) * 3;

    const int grayIdx = y * 32 + x;

    const float r = static_cast<float>(colorInput[colorIdx + 0]);
    const float g = static_cast<float>(colorInput[colorIdx + 1]);
    const float b = static_cast<float>(colorInput[colorIdx + 2]);

    const float gray = 0.299f * r + 0.587f * g + 0.114f * b;

    grayOutput[grayIdx] = static_cast<uint8_t>(gray);
}

int main()
{
    // 32×32 컬러 이미지 로드 (3채널)
    int imgWidth, imgHeight, imgChannels;
    uint8_t* hostColorImage = stbi_load("cat32color.png",
        &imgWidth, &imgHeight, &imgChannels, 3); 
    
    assert(imgWidth == 32 && imgHeight == 32 && imgChannels == 3);
    
    uint8_t* deviceColorInput;
    uint8_t* deviceGrayOutput;
    cudaMalloc(&deviceColorInput, 32 * 32 * 3 * sizeof(uint8_t)); 
    cudaMalloc(&deviceGrayOutput, 32 * 32 * sizeof(uint8_t));     
    
    cudaMemcpy(deviceColorInput, hostColorImage, 32 * 32 * 3, cudaMemcpyHostToDevice);
    
    constexpr dim3 blockSize(32, 32); 
    colorToGrayscaleKernel<<<1, blockSize>>>(deviceColorInput, deviceGrayOutput);
    
    cudaDeviceSynchronize();
    
    // 흑백 결과를 CPU로 복사
    std::vector<uint8_t> hostGrayResult(32 * 32);
    cudaMemcpy(hostGrayResult.data(), deviceGrayOutput, 32 * 32, cudaMemcpyDeviceToHost);
    
    // 흑백 이미지 저장
    stbi_write_png("cat32gray_converted.png", 32, 32, 1, hostGrayResult.data(), 32);
    
    // 메모리 해제
    cudaFree(deviceColorInput);
    cudaFree(deviceGrayOutput);
    stbi_image_free(hostColorImage);
    
    std::cout << "Color to grayscale conversion completed!" << std::endl;
    return 0;
}