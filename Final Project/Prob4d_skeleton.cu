#include <iostream>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

using namespace std;

__host__ __device__ float func(float x)
{
  // Implement the function evaluation at point x
  // This function will be called both by the gpu and the cpu
  // Recall the trapezoidal rule correction
}

__global__ void trinteg(float* vals, const float stepsize, const unsigned int nsteps)
{
  // compute the funcion evaluation coordinate
  // based on the blockIdx.x blockDim.x and threadIdx.x
  // assign the function value to the proper locatiion in vals
  // make sure you don't exceed the dimension of vals
}

float evalfunc(const float stepsize, const unsigned int step)
{
  // compute the step size
}

double reduce_vector(float* vals, const unsigned int vlen)
{
  thrust::device_vector<float> dsums(vlen);
  float *dptr = thrust::raw_pointer_cast(&dsums[0]);
  cudaMemcpy(dptr, vals, vlen*sizeof(float), cudaMemcpyDeviceToDevice);

  double sum = thrust::reduce(dsums.begin(),dsums.end());
  return sum;
}

int main(int argc, char* argv[])
{ 
  const unsigned int default_nsteps = 1000000;
  const unsigned int default_nthreads = 256;
  unsigned int nsteps_param = 0;
  unsigned int nthreads_param = 0;
  
  for (unsigned int k=1; k<argc; )
    {
    if (strcmp(argv[k], "--steps") == 0)
      {
      if (argc<k+2)
	{
        printf("error: --steps has to be followed by a number\n");
	exit(1);
	}
      else
        {
	nsteps_param = atoi(argv[k+1]);
        }
      k+=2;
      continue;
      }
    if (strcmp(argv[k], "--threads") == 0)
      {
      if (argc<k+2)
	{
        printf("error: --threads has to be followed by a number\n");
	exit(1);
	}
      else
        {
	nthreads_param = atoi(argv[k+1]);
        }
      k+=2;
      continue;
      }
    {
    printf("error: %s unrecognized keyword\n", argv[k]);
    exit(1);
    }
    }
  std::cout.precision(std::numeric_limits<double>::max_digits10-1);
  const double lower_limit = 0.e+00;
  const double upper_limit = M_PI/2.e+00;

  const unsigned int nsteps = nsteps_param ? nsteps_param : default_nsteps;
  const unsigned int nthreads = nthreads_param ? nthreads_param : default_nthreads;
  const unsigned int nblocks = ?;
  const float stepsize = (upper_limit-lower_limit)/(nsteps-1);

  float* vals_dev = NULL;
  // allocate vals_dev on the GPU (proper cudaMalloc
  
  auto start = std::chrono::high_resolution_clock::now();
  trinteg<<< ? , ?>>>( , , );
  
  double sum = reduce_vector(vals_dev, nsteps);

  // trapezoidal rule end-point correction
  sum -= ?
  sum *= // don't forget to multiply by the stepsize
  auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed_seconds{end - start};
  cout << "running time on gpu float type: " << elapsed_seconds.count() << endl;
  cout << "integ: " << sum << endl;

  return 0;
}
