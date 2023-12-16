#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <math.h>

using namespace std;

// Function declarations
template <typename T>
T cei(T k, const unsigned int terms);

template <typename T>
T func(T x);

template <typename T>
T evalfunc(T stepsize, const unsigned int step);

template <typename T>
T trinteg(T stepsize, const unsigned int nsteps);

// Function definitions
template <typename T>
T doubleFactorial(int n) {
    if (n <= 0) {
        return T(1.0);
    }

    T result = T(1.0);
    for (int i = n; i > 0; i -= 2) {
        result *= T(i);
    }

    return result;
}

template <typename T>
T ellipticIntegral(T k, int terms) {
    T result = T(0.5) * M_PI;

    T term = T(1.0);
    T kSquared = k * k;

    for (int n = 1; n <= terms; n++) {
        T new_term = (doubleFactorial<T>(2 * n - 1) / doubleFactorial<T>(2 * n)) *
                     (doubleFactorial<T>(2 * n - 1) / doubleFactorial<T>(2 * n)) *
                     pow(kSquared, n) / T(2 * n - 1);
        term -= new_term;
    }

    return result * term;
}

template <typename T>
T cei(T k, const unsigned int terms) {
    // Call the ellipticIntegral function
    return ellipticIntegral<T>(k, terms);
}

template <class T>
T evalfunc(const T stepsize, const unsigned int step) {
    T xi = step * stepsize;
    return func(xi);
}

template <typename T>
T evalfunc(T stepsize, const unsigned int step) {
    return func(step * stepsize);
}

template <typename T>
T trinteg(T stepsize, const unsigned int nsteps) {
    T sum = 0.0;

    // OpenMP parallelization for the reduction
    #pragma omp parallel for reduction(+:sum)
    for (unsigned int i = 0; i < nsteps; ++i) {
        T xi = i * stepsize;
        T fi = evalfunc(stepsize, i);

        if (i == 0 || i == nsteps - 1) {
            sum += 0.5 * fi;
        } else {
            sum += fi;
        }
    }

    return sqrt(2.0) * stepsize * sum;
}

int main(int argc, char* argv[]) {
    const unsigned int default_nsteps = 1000000;
    const unsigned int default_nthreads = 4;
    unsigned int nsteps_param = 0;
    unsigned int nthreads_param = 0;

    for (unsigned int k = 1; k < argc;) {
        if (strcmp(argv[k], "--steps") == 0) {
            if (argc < k + 2) {
                printf("error: --steps has to be followed by a number\n");
                exit(1);
            } else {
                nsteps_param = atoi(argv[k + 1]);
            }
            k += 2;
            continue;
        }
        if (strcmp(argv[k], "--threads") == 0) {
            if (argc < k + 2) {
                printf("error: --threads has to be followed by a number\n");
                exit(1);
            } else {
                nthreads_param = atoi(argv[k + 1]);
            }
            k += 2;
            continue;
        } else {
            printf("error: %s unrecognized keyword\n", argv[k]);
            exit(1);
        }
    }

    std::cout.precision(std::numeric_limits<double>::max_digits10 - 1);

    // Call cei to get the reference value
    double len = cei<double>(std::sqrt(2.0) / 2.0, 100);

    // Define variables
    double lower_limit = 0.0;
    double upper_limit = M_PI / 2.0;
    double step_size = (upper_limit - lower_limit) / (nsteps_param ? nsteps_param : default_nsteps);
    double intval = 0.0;

    const unsigned int nsteps = nsteps_param ? nsteps_param : default_nsteps;
    const unsigned int nthreads = nthreads_param ? nthreads_param : default_nthreads;
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);

    auto start = std::chrono::high_resolution_clock::now();
    intval = trinteg(step_size, nsteps);
    auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Integration uses " << nsteps << " steps" << std::endl;
    std::cout << "cei1:  " << std::scientific << len << std::endl;
    std::cout << "integ: " << intval << std::endl;
    std::cout << "running time on " << nthreads << " threads: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}