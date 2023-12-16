#include <iostream>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using namespace std;

// Function to evaluate the integrand
__host__ __device__ float func(float x)
{
    // Implement the function evaluation at point x
    // This function will be called both by the GPU and the CPU
    // Recall the trapezoidal rule correction
    return sqrt(1 - 0.5 * pow(sin(x), 2));
}

// Kernel function to compute the function values at each step
__global__ void trinteg(float* vals, const float stepsize, const unsigned int nsteps)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nsteps)
    {
        float xi = idx * stepsize;
        vals[idx] = func(xi);
    }
}

// Function to evaluate the function at a specific step
float evalfunc(const float stepsize, const unsigned int step)
{
    float xi = step * stepsize;
    return func(xi);
}

// Function to reduce the GPU vector using thrust
double reduce_vector(float* vals, const unsigned int vlen)
{
    thrust::device_vector<float> dsums(vals, vals + vlen);
    double sum = thrust::reduce(dsums.begin(), dsums.end());
    return sum;
}

int main(int argc, char* argv[])
{
    const unsigned int default_nsteps = 1000000;
    const unsigned int default_nthreads = 256;
    unsigned int nsteps_param = 0;
    unsigned int nthreads_param = 0;

    for (unsigned int k = 1; k < argc;)
    {
        if (strcmp(argv[k], "--steps") == 0)
        {
            if (argc < k + 2)
            {
                printf("error: --steps has to be followed by a number\n");
                exit(1);
            }
            else
            {
                nsteps_param = atoi(argv[k + 1]);
            }
            k += 2;
            continue;
        }
        if (strcmp(argv[k], "--threads") == 0)
        {
            if (argc < k + 2)
            {
                printf("error: --threads has to be followed by a number\n");
                exit(1);
            }
            else
            {
                nthreads_param = atoi(argv[k + 1]);
            }
            k += 2;
            continue;
        }
        else
        {
            printf("error: %s unrecognized keyword\n", argv[k]);
            exit(1);
        }
    }

    std::cout.precision(std::numeric_limits<double>::max_digits10 - 1);
    const double lower_limit = 0.0;
    const double upper_limit = M_PI / 2.0;

    const unsigned int nsteps = nsteps_param ? nsteps_param : default_nsteps;
    const unsigned int nthreads = nthreads_param ? nthreads_param : default_nthreads;
    const unsigned int nblocks = (nsteps + nthreads - 1) / nthreads;
    const float stepsize = (upper_limit - lower_limit) / (nsteps - 1);

    float* vals_dev = nullptr;
    cudaMalloc((void**)&vals_dev, nsteps * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    trinteg<<<nblocks, nthreads>>>(vals_dev, stepsize, nsteps);

    cudaDeviceSynchronize(); // Ensure all GPU computations are finished

    double sum = reduce_vector(vals_dev, nsteps);

    // Trapezoidal rule end-point correction
    sum -= 0.5 * (func(lower_limit) + func(upper_limit));
    sum *= stepsize;

    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;
    cout << "running time on GPU float type: " << elapsed_seconds.count() << " seconds" << endl;
    cout << "integ: " << sum << endl;

    // Free allocated GPU memory
    cudaFree(vals_dev);

    return 0;
}