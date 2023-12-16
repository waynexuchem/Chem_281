#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <string.h>

using namespace std;

__global__
void saxpy(int n, float alpha, float* x, float* y)
{
  unsigned int pos  = blockIdx.x*blockDim.x + threadIdx.x;
  if (pos < n) y[pos] += alpha*x[pos];
}

int main(int argc, char* argv[])
{
  std::random_device rd;
  std::mt19937 rg(rd());
  std::uniform_real_distribution<> genval(0, 12);
  unsigned int dim = 1024*1024*1024;
  const unsigned int blockSize = 256;
  const double alpha = 1.4f;
  float* x = new float[dim];
  float* y = new float[dim];

  float* result = new float[dim];

  float* device_x = NULL;
  float* device_y = NULL;

  cudaMalloc(&device_x, dim*sizeof(float));
  cudaMalloc(&device_y, dim*sizeof(float));

  // Initialize arrays
  for (unsigned int i=0; i<dim; i++)
    {
    x[i] = genval(rg);
    y[i] = genval(rg);
    }

  // copy data to device
  cudaMemcpy(device_x, x, dim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_y, y, dim*sizeof(float), cudaMemcpyHostToDevice);

  auto start = std::chrono::high_resolution_clock::now();
  saxpy<<<(dim+blockSize-1)/blockSize, blockSize>>>(dim, alpha, device_x, device_y);
  auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed_seconds{end - start};
  cout << "running time: " << elapsed_seconds.count() << endl; 

  cudaMemcpy(result, device_y, dim*sizeof(float), cudaMemcpyDeviceToHost);



  start = std::chrono::high_resolution_clock::now();

// Here implement the cpu version. Compare execution time

  end = std::chrono::high_resolution_clock::now();
  {
  const std::chrono::duration<double> elapsed_seconds{end - start};
  cout << "cpu running time: " << elapsed_seconds.count() << endl; 
  }

// Verification - check that cpu and gpu results match
/* Uncomment these lines
  double error = 0.0e+00;
  for (unsigned int i=0; i<dim; i++)
    error += std::abs(result[i]-y[i]);

  cout << "gpu-cpu error " << error << endl;
*/
  cudaFree(device_x);
  cudaFree(device_y);

  delete[] result;
  delete[] y;
  delete[] x;
  
  return 0;
}
