#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

template <class T>
T func(const T x) {
    return sqrt(1 - 0.5 * pow(sin(x), 2));
}

template <class T>
T evalfunc(const T stepsize, const unsigned int step) {
    T xi = step * stepsize;
    return func(xi);
}

template <class T>
T trinteg(const unsigned int n, const T a, const T b) {
    const T h = (b - a) / (n - 1);
    T sum = 0.0;

    // Trapezoidal rule
    for (unsigned int i = 0; i < n; ++i) {
        T xi = a + i * h;
        T fi = func(xi);

        if (i == 0 || i == n - 1) {
            // Include the endpoints with a weight of 0.5
            sum += 0.5 * fi;
        } else {
            // Interior points
            sum += fi;
        }
    }

    // Multiply by h for the final result
    return sqrt(2.0) * h * sum;
}

int main() {
    const unsigned int nsteps = 1000000;  // Number of steps

    auto start_double = chrono::high_resolution_clock::now();
    double result_double = trinteg<double>(nsteps, 0.0, M_PI / 2.0);
    auto stop_double = chrono::high_resolution_clock::now();
    auto duration_double = chrono::duration_cast<chrono::microseconds>(stop_double - start_double);

    auto start_float = chrono::high_resolution_clock::now();
    float result_float = trinteg<float>(nsteps, 0.0f, M_PI / 2.0f);
    auto stop_float = chrono::high_resolution_clock::now();
    auto duration_float = chrono::duration_cast<chrono::microseconds>(stop_float - start_float);

    cout << "Length of the quarter arc of sin(x) (double): " << result_double << endl;
    cout << "Execution time (double): " << duration_double.count() << " microseconds" << endl;

    cout << "Length of the quarter arc of sin(x) (float): " << result_float << endl;
    cout << "Execution time (float): " << duration_float.count() << " microseconds" << endl;

    return 0;
}
