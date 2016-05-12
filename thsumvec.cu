#include <cuda.h>
#include <math.h>
#include <iostream>
#include <thrust/device_vector.h>


static const size_t N = 102400;

__global__ void kernel(const thrust::device_ptr<float> A, const thrust::device_ptr<float> B, thrust::device_ptr<float> C, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    thrust::device_vector<float> d_A, d_B, d_C;
    d_A.resize(N);
    d_B.resize(N);
    d_C.resize(N);

    for (int i = 0; i < N; i++) {
      d_A[i] = i;
      d_B[i] = 0.5f * i - 2;
    }

    kernel<<<ceil(double(N) / 512), 512>>>(d_A.data(), d_B.data(), d_C.data(), N);

    double err = 0;
    for (int i = 0; i < N; i++) {
      err += (d_A[i] + d_B[i]) - d_C[i];
    }
    std::cout << "Cum error: " << sqrt(err) << std::endl;
    return 0;
}
