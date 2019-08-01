import sys
import os
import ctypes
import platform


if platform.system() == 'Linux':
	cur_path = sys.path[0]
	dll_path = os.path.join(cur_path, "tensorfly", "core.so")
	c_core = ctypes.CDLL(dll_path)
else:
    cur_path = os.path.dirname(__file__)
    dll_path = os.path.join(cur_path, "tensorfly", "core.dll")
    c_kernel = ctypes.CDLL(dll_path)
