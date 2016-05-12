#include <iostream>
#include <cuda.h>

int main(int argc, char **argv) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    std::cout << "name: " << devProp.name << std::endl;
    std::cout << "l2CacheSize: " << devProp.l2CacheSize << std::endl;
    std::cout << "regsPerBlock: " << devProp.regsPerBlock << std::endl;
    std::cout << "regsPerMultiprocessor: " << devProp.regsPerMultiprocessor << std::endl;
    std::cout << "unifiedAddressing: " << devProp.unifiedAddressing << std::endl;
    std::cout << "warpSize: " << devProp.warpSize << std::endl;
    std::cout << "CC: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "maxThreadsDim: " << devProp.maxThreadsDim[0] << std::endl;
    std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxGridSize: " << devProp.maxGridSize[0] << "x" << devProp.maxGridSize[1] << "x" << devProp.maxGridSize[2] << std::endl;
    std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock << std::endl;
    std::cout << "sharedMemPerMultiprocessor: " << devProp.sharedMemPerMultiprocessor << std::endl;
    return 0;
}
