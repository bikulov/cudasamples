#include <cuda.h>
#include <math.h>
#include <iostream>
#include <vector>

static const size_t N = 102400;

__global__ void kernel(const float* A, const float* B, float* C, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    std::vector<float> h_A, h_B, h_C;
    h_A.resize(N);
    h_B.resize(N);
    h_C.resize(N);

    for (int i = 0; i < N; i++) {
      h_A[i] = i;
      h_B[i] = 0.5f * i - 2;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);

    cudaMemcpy(d_A, &h_A[0], sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &h_B[0], sizeof(float) * N, cudaMemcpyHostToDevice);

    kernel<<<512, ceil(double(N) / 512)>>>(d_A, d_B, d_C, N);

    cudaMemcpy(&h_C[0], d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);
  
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    double err = 0;
    for (int i = 0; i < N; i++) {
      err += (h_A[i] + h_B[i]) - h_C[i];
    }
    std::cout << "Cum error: " << sqrt(err) << std::endl;
    return 0;
}
