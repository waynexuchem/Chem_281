#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

double doubleFactorial(int n) {
    if (n <= 0) {
        return 1.0;
    }

    double result = 1.0;
    for (int i = n; i > 0; i -= 2) {
        result *= i;
    }

    return result;
}

double ellipticIntegral(double k, int terms) {
    double result = 0.5 * M_PI;

    double term = 1.0;
    double kSquared = k * k;

    for (int n = 1; n <= terms; n++) {
        double new_term = (doubleFactorial(2 * n - 1) / doubleFactorial(2 * n)) * (doubleFactorial(2 * n - 1) / doubleFactorial(2 * n)) * pow(kSquared, n) / (2 * n - 1);
        // cout << n << " " << doubleFactorial(2 * n - 1) << " " << doubleFactorial(2 * n) << " " << pow(kSquared, n) << " " << 2 * n - 1 << " " << new_term << endl;
        // cout << "Term " << n << ": " << new_term << endl;
        term -= new_term;
    }
    // cout << "Result: " << result * term << endl;
    return result * term;
}

int main() {
    double k = sqrt(2.0) / 2.0;

    int terms = 30;
    double prevLength = numeric_limits<double>::infinity();
    double length = sqrt(2.0) * ellipticIntegral(k, terms);
    cout << "Length of the quarter arc of sin(x): " << length << endl;

    // Iteratively increase the number of terms until convergence
    while (abs(length - prevLength) > 1e-16) {
        prevLength = length;
        terms *= 2; // Double the number of terms
        length = sqrt(2.0) * ellipticIntegral(k, terms);
    }

    cout << "Number of terms: " << terms << endl;
    cout << "Length of the quarter arc of sin(x): " << length << endl;

    // Multiply by 4 for the entire cycle
    length *= 4;

    cout << "Length of the entire cycle of sin(x): " << length << endl;

    return 0;
}