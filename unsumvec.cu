#include <cuda.h>
#include <math.h>
#include <iostream>

static const size_t N = 102400;

__global__ void kernel(const float* A, const float* B, float* C, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    float *h_A, *h_B, *h_C;

    cudaHostAlloc(&h_A, sizeof(float) * N, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, sizeof(float) * N, cudaHostAllocDefault);
    cudaHostAlloc(&h_C, sizeof(float) * N, cudaHostAllocDefault);
    
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 0.5f * i - 2;
    }

    float *d_A, *d_B, *d_C;
    cudaHostGetDevicePointer(&d_A, h_A, 0);
    cudaHostGetDevicePointer(&d_B, h_B, 0);
    cudaHostGetDevicePointer(&d_C, h_C, 0);

    kernel<<<ceil(double(N) / 512), 512>>>(d_A, d_B, d_C, N);
    cudaThreadSynchronize();
  
    double err = 0;
    for (int i = 0; i < N; i++) {
        err += (h_A[i] + h_B[i]) - h_C[i];
    }
    std::cout << "Cum error: " << sqrt(err) << std::endl;
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
