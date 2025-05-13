#include <iostream>
#include <vector>
#include <chrono>

int main() {
    constexpr uint64_t num_elements = 1'000'000'000;

    std::cout << "Allocating vector with " << num_elements << " float elements (~4GB)..." << std::endl;
    std::vector<float> data(num_elements, 1.0f);

    const auto start = std::chrono::high_resolution_clock::now();
    for(auto& element : data)
    {
        element = element * 10.0f;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;

    std::cout << "First and last element: " << data[0] << " " << data[num_elements - 1] << std::endl;
    std::cout << "Computation completed in " << total_time.count() << " seconds" << std::endl;
    std::cout << "Total time: " << total_time.count() << " seconds" << std::endl;

    return 0;
}