#include "gridsample.h"
#include <iostream>
int main() {
    DgGridSample sample(0,0,0,nullptr,nullptr,nullptr);
    sample.Compute();
    std::cout << "Hello, World!" << std::endl;
    return 0;
}