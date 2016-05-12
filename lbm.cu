#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <vector>

#define QNUM 19

__constant__ int NX;
__constant__ int NY;
__constant__ int NZ;
__constant__ int SIZE;

__constant__ float WW[QNUM];
__constant__ char CX[QNUM];
__constant__ char CY[QNUM];
__constant__ char CZ[QNUM];
__constant__ char OPPQ[QNUM];

static const char Cx [] = {0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, 1, -1, -1, 0, 0, 0, 0};
static const char Cy [] = {0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, -1, 1};
static const char Cz [] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, 1, -1, -1};
static const float W [] = {
    12.0 / 36.0,
    2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
static const char OppQ [] = {0, 2, 1, 4, 3, 6, 5, 9, 10, 7, 8, 13, 14, 11, 12, 17, 18, 15, 16};

void Init(unsigned char *raw, float *f, float *rho, float *ux, float *uy, float *uz, int nx, int ny, int nz) {
    cudaMemcpyToSymbol(NX, &nx, sizeof(int));
    cudaMemcpyToSymbol(NY, &ny, sizeof(int));
    cudaMemcpyToSymbol(NZ, &nz, sizeof(int));
    
    const size_t size = nx * ny * nz; 
    cudaMemcpyToSymbol(SIZE, &size, sizeof(size_t));
    
    cudaMemcpyToSymbol(CX, &Cx, QNUM * sizeof(char));
    cudaMemcpyToSymbol(CY, &Cy, QNUM * sizeof(char));
    cudaMemcpyToSymbol(CZ, &Cz, QNUM * sizeof(char));
    cudaMemcpyToSymbol(WW, &W,  QNUM * sizeof(float));
    cudaMemcpyToSymbol(OPPQ, &OppQ, QNUM * sizeof(char));

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                size_t index = x + y * nx + z * nx * ny;

                if (raw[index] == 0) {
                    for (size_t q = 0; q < QNUM; ++q) {
                        float cu = (Cx[q] * ux[index] + Cy[q] * uy[index] + Cz[q] * uz[index]);
                        float u2 = ux[index] * ux[index] + uy[index] * uy[index] + uz[index] * uz[index];
                        float feq = W[q] * rho[index] * (1 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                        f[q * size + index] = feq;
                    }
                }
            }
        }
    }
}

__global__ void SRTKernel(unsigned char *raw, float *f, float *fNew, float tau = 1.0) {
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t z = threadIdx.z + blockIdx.z * blockDim.z;
    size_t index = x + y * NX + z * NX * NY;
    float localRho = 0;
    float localUx = 0;
    float localUy = 0;
    float localUz = 0.00001;
    if (raw[index] == 0) {
        for (size_t q = 0; q < QNUM; ++q) {
            localRho += f[q * SIZE + index];
            localUx += CX[q] * f[q * SIZE + index];
            localUy += CY[q] * f[q * SIZE + index];
            localUz += CZ[q] * f[q * SIZE + index];
        }

        for (size_t q = 0; q < QNUM; ++q) {
            int newX = (x + CX[q] + NX) % NX;
            int newY = (y + CY[q] + NY) % NY;
            int newZ = (z + CZ[q] + NZ) % NZ;
            size_t newIndex = newX + newY * NX + newZ * NX * NY;

            float cu = (CX[q] * localUx + CY[q] * localUy + CZ[q] * localUz);
            float u2 = localUx * localUx + localUy * localUy + localUz * localUz;
            float feq = WW[q] * localRho * (1 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
            float tmpF = f[q * SIZE + index] + (1.0 / tau) * (feq - f[q * SIZE + index]);

            if (raw[newIndex] == 0) {
                fNew[q * SIZE + newIndex] = tmpF;
            } else {
                fNew[OPPQ[q] * SIZE + index] = tmpF;
            }
        }
    } else {
        for (size_t q = 0; q < QNUM; ++q) {
             fNew[q * SIZE + index] = 0;
        }
    }
}

int main(int /*argc*/, char** /*argv*/) {
    const int nx = 128;
    const int ny = 128;
    const int nz = 128;
    const int size = nx * ny * nz;
    std::vector<float> rho, ux, uy, uz, f;
    rho.resize(size);
    ux.resize(size);
    uy.resize(size);
    uz.resize(size);
    f.resize(QNUM * size);

    std::vector<unsigned char> raw;
    raw.resize(size);

    const int R = 50;
    int index = 0;

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                index = x + y * nx + z * nx * ny;
                ux[index] = uy[index] = uz[index] = 0;
                if ((x-nx/2)*(x-nx/2) + (y-ny/2)*(y-ny/2) < R*R) {
                    raw[index] = 0;
                    rho[index] = 1.f;
                } else {
                    raw[index] = 1;
                    rho[index] = 0.f;
                }
            }
        }
    }

    Init(&raw[0], &f[0], &rho[0], &ux[0], &uy[0], &uz[0], nx, ny, nz);

    float *d_f, *d_fNew;
    cudaMalloc(&d_f, sizeof(float) * QNUM * size);
    cudaMalloc(&d_fNew, sizeof(float) * QNUM * size);
    cudaMemcpy(d_f, &f[0], sizeof(float) * QNUM * size, cudaMemcpyHostToDevice);
    
    unsigned char *d_raw;
    cudaMalloc(&d_raw, sizeof(unsigned char) * size);
    cudaMemcpy(d_raw, &raw[0], sizeof(unsigned char) * size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 4, 4);
    dim3 blocksPerGrid(nx/threadsPerBlock.x, ny/threadsPerBlock.y, nz/threadsPerBlock.z);
   
    float *fTmp = NULL; 
    for (int it = 0; it < 1000; ++it) {
        SRTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_raw, d_f, d_fNew);
        fTmp = d_f;
        d_f = d_fNew;
        d_fNew = fTmp;
    }
    
    cudaMemcpy(&f[0], d_f, sizeof(float) * QNUM * size, cudaMemcpyDeviceToHost);

    for (int x = 0; x < nx; ++x) {
        int y = ny / 2;
        int z = nz / 2;
        float locUz = 0;
        for (size_t q = 0; q < QNUM; ++q) {
            locUz += Cz[q] * f[q * size + x + y * nx + z * nx * ny];
        }
        std::cout << locUz << std::endl;
    }

    cudaFree(d_f);
    cudaFree(d_fNew);
    cudaFree(d_raw);
    
    return 0;
}

