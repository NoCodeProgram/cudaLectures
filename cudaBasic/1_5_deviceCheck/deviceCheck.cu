#include <iostream>
#include <cuda_runtime.h>

// Function to get CUDA cores per SM based on compute capability
int getCoresPerSM(int major, int minor) {
    int cores = 0;
    switch ((major << 4) + minor) {
        case 0x10: cores = 8; break;   // Tesla
        case 0x11: cores = 8; break;
        case 0x12: cores = 8; break;
        case 0x13: cores = 8; break;
        case 0x20: cores = 32; break;  // Fermi
        case 0x21: cores = 48; break;
        case 0x30: cores = 192; break; // Kepler
        case 0x32: cores = 192; break;
        case 0x35: cores = 192; break;
        case 0x37: cores = 192; break;
        case 0x50: cores = 128; break; // Maxwell
        case 0x52: cores = 128; break;
        case 0x53: cores = 128; break;
        case 0x60: cores = 64; break;  // Pascal
        case 0x61: cores = 128; break;
        case 0x62: cores = 128; break;
        case 0x70: cores = 64; break;  // Volta
        case 0x72: cores = 64; break;
        case 0x75: cores = 64; break;  // Turing
        case 0x80: cores = 64; break;  // Ampere
        case 0x86: cores = 128; break;
        case 0x87: cores = 128; break;
        case 0x89: cores = 128; break; // Ada Lovelace
        case 0x90: cores = 128; break; // Hopper
        default: cores = 0; break;
    }
    return cores;
}

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    std::cout << "CUDA Block Configuration Information" << std::endl;
    std::cout << "====================================" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        int coresPerSM = getCoresPerSM(prop.major, prop.minor);
        int totalCores = coresPerSM * prop.multiProcessorCount;
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << " Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << std::endl;
        
        // SM Information
        std::cout << " Streaming Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
        std::cout << " CUDA Cores per SM: " << coresPerSM << std::endl;
        std::cout << " Total CUDA Cores: " << totalCores << std::endl;
        std::cout << std::endl;
        
        // Block Configuration
        std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << " Max Thread Dimensions: (" << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << " Warp Size: " << prop.warpSize << std::endl;
        std::cout << " Max Warps per Block: " << prop.maxThreadsPerBlock / prop.warpSize << std::endl;
        std::cout << std::endl;
        
        // SM Resource Limits
        std::cout << " Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << " Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << " Max Warps per SM: " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;
        std::cout << std::endl;
        
        // Memory per Block
        std::cout << " Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << " Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << " Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << " Registers per SM: " << prop.regsPerMultiprocessor << std::endl;
        std::cout << std::endl;
        
        // Occupancy Calculations
        std::cout << " Theoretical Occupancy:" << std::endl;
        std::cout << "  - Based on threads: " << prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock << " blocks per SM" << std::endl;
        std::cout << "  - Based on block limit: " << prop.maxBlocksPerMultiProcessor << " blocks per SM" << std::endl;
        
        std::cout << "====================================" << std::endl;
    }
    
    return 0;
}