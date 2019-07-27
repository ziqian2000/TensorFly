import sys
import os
import ctypes

cur_path = sys.path[0]
dll_path = os.path.join(cur_path, "tensorfly", "core.so")
c_core = ctypes.CDLL(dll_path)