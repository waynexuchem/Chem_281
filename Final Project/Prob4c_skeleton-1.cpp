#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <math.h>

using namespace std;

/*
Note:
c++ versions prior to 20 don't have pi defined
<math.h> is included for the definiton of M_PI
for compilers implementing c++ 20 include <numbers> and use std::pi
 */


<function declaration> cei(<some type> k, const unsigned int terms)
{
  // k is the parameter of the complete elliptical integral of the second kind
  // evaluate the complete elliptical integral series.
  return result;
}

<function declaration> func(<some type> x)
{
  // compute the value of the integrand at position x
}

<function declaration> evalfunc(<some type> stepsize, const unsigned int step)
{
  // compute the function evaluation coordinate
  return func at x;
}

<function declaration trinteg(<some type> stepsize, const unsigned int nsteps)
{

#pragma omp ?
  // compute the sum of the function evaluations at every step
  // apply trapezoidal rule end-point correction
  return ?;
}

int main(int argc, char* argv[])
{
  const unsigned int default_nsteps = 1000000;
  const unsigned int default_nthreads = 4;
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
  call cei to get the reference value

  <?> len = cei(<?>, <?>);

  <?> lower_limit = 0.e+00;
  <?> upper_limit = M_PI/2.e+00;

  const unsigned int nsteps = nsteps_param ? nsteps_param : default_nsteps;
  const unsigned int nthreads = nthreads_param ? nthreads_param : default_nthreads;
  omp_set_dynamic(0);
  omp_set_num_threads(nthreads);
  
  <?> step_size = (upper_limit-lower_limit)/(nsteps-1);
  auto start = std::chrono::high_resolution_clock::now();
  <?> intval = trinteg(step_size, nsteps);
  auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed_seconds{end - start};
  cout << "Integration uses " << nsteps << " steps" << endl;
  cout << "cei1:  " << std::scientific << <?> << endl;
  cout << "integ: " << intval << endl;
  cout << "running time on " << nthreads << " threads << ": " << elapsed_seconds.count() << endl; 
  return 0;
}
