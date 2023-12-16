#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

double func(const double x) {
    return sqrt(1 - 0.5 * pow(sin(x), 2));
}

double evalfunc(const double stepsize, const unsigned int step) {
    double xi = step * stepsize;
    return func(xi);
}

double trinteg(const unsigned int n, const double a, const double b) {
    const double h = (b - a) / (n - 1);
    double sum = 0.0;

    // Trapezoidal rule
    for (unsigned int i = 0; i < n; ++i) {
        double xi = a + i * h;
        double fi = func(xi);

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
    const double a = 0.0;  // Lower bound
    const double b = M_PI / 2.0;  // Upper bound

    auto start = chrono::high_resolution_clock::now();

    // Perform the trapezoidal integration
    double result = trinteg(nsteps, a, b);

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Length of the quarter arc of sin(x): " << result << endl;
    cout << "Execution time: " << duration.count() << " microseconds" << endl;

    return 0;
}