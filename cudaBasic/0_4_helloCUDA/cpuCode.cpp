#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>

int main() {
    constexpr uint64_t numElements = 2'000'000'000;

    std::cout << "Allocating vector with " << numElements << " float elements (~8GB)..." << std::endl;
    
    // Modern C++ random number generation
    constexpr uint64_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);
    
    // Initialize vector with random values
    std::vector<float> data;
    data.reserve(numElements);
    
    std::cout << "Initializing with random values..." << std::endl;
    for (uint64_t i = 0; i < numElements; ++i) {
        data.emplace_back(dist(gen));
    }

    std::cout << "Starting computation..." << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();
    
    for (auto& element : data) {
        element *= 10.0f;
    }
    
    const auto end = std::chrono::high_resolution_clock::now();
    const auto totalTime = std::chrono::duration<double>(end - start);
    
    std::cout << "First and last element: " << data[0] << " " << data[numElements - 1] << std::endl;
    std::cout << "Computation completed in " << totalTime.count() << " seconds" << std::endl;

    return 0;
}