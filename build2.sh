gcc -o tensorfly/core.so -shared -fPIC -fopenmp tensorfly/core.cpp -lopenblas -O4