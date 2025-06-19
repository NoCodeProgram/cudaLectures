#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    std::cout << "Number of CUDA devices found: " << deviceCount << std::endl;
    std::cout << "=================================" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << std::fixed << std::setprecision(2);
        
        // Basic Device Information
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << " Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        // Memory Information
        std::cout << " Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB (" << prop.totalGlobalMem / (1024.0*1024*1024) << " GB)" << std::endl;
        std::cout << " Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << " L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << " Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << " Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        
        // Execution Configuration
        std::cout << " Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << " Warp Size: " << prop.warpSize << std::endl;
        std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << " Max Thread Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << " Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        
        // Performance and Architecture
        std::cout << " Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << " Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        
        // Key Features
        std::cout << " Concurrent Kernels: " << (prop.concurrentKernels ? "Supported" : "Not Supported") << std::endl;
        std::cout << " ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << " Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << " Managed Memory: " << (prop.managedMemory ? "Yes" : "No") << std::endl;
        
        // System Information
        std::cout << " Integrated Memory: " << (prop.integrated ? "Yes" : "No") << std::endl;
        std::cout << " PCI Bus ID: " << prop.pciBusID << std::endl;
        std::cout << " PCI Device ID: " << prop.pciDeviceID << std::endl;
        
        // Calculate memory bandwidth
        float memoryBandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        std::cout << " Memory Bandwidth: " << memoryBandwidth << " GB/s" << std::endl;
        
        std::cout << "=================================" << std::endl;
    }
    
    return 0;
}