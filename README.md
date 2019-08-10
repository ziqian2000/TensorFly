# TensorFly

TensorFly is an open source project (for SJTU  PPCA 2019 ML-system). Its [API](<https://tensorflow.google.cn/versions/r1.14/api_docs/python/tf>) is almost the same as TensorFlow (r1.14 stable), but many methods are not implemented.

## How to run test

Testcases are [here](<https://github.com/dizhenhuoshan/ML-System-2019-Testcase>).

1. Download **numpy**.

2. Download **Intel Math Kernel Library**. You can follow the instructions [here](<https://blog.csdn.net/chenjun15/article/details/75041932/>).

3. To compile `core.c`:

   ```bash
   gcc -o tensorfly/core.so -shared -fPIC -fopenmp tensorfly/core.c -lmkl_rt -O4 -funroll-loops
   ```

4. Run test:

   ```bash
   python3 run_test.py tensorfly
   ```

