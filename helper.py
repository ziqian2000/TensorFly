import numpy as np
import ctypes

def find_topo_sort(node_list):
	"""Given a list of nodes, return a topological sort list of nodes ending in them.
	
	A simple algorithm is to do a post-order DFS traversal on the given nodes, 
	going backwards based on input edges. Since a node is added to the ordering
	after all its predecessors are traversed due to post-order DFS, we get a topological
	sort.
	"""
	visited = set()
	topo_order = []
	for node in node_list:
		topo_sort_dfs(node, visited, topo_order)
	return topo_order

def topo_sort_dfs(node, visited, topo_order):
	"""Post-order DFS"""
	if node in visited:
		return
	visited.add(node)
	for n in node.inputs:
		topo_sort_dfs(n, visited, topo_order)
	topo_order.append(node)

def sum_node_list(node_list):
	"""Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
	from operator import add
	from functools import reduce
	return reduce(add, node_list)

def zero_padding_expand(tensor, up, down, left, right):
	"""expand dimension 1 and 2 with zero padding"""
	n_tensor = np.zeros([tensor.shape[0], tensor.shape[1] + up + down, tensor.shape[2] + left + right, tensor.shape[3]], dtype = tensor.dtype)
	n_tensor[:, up : up + tensor.shape[1], left : left + tensor.shape[2], :] = tensor[:, :, :, :]
	return n_tensor

def calc_new_len(h1, h2, stride): # h1 >= h2
	t = h1 // stride
	if(h1 % stride != 0): t += 1
	return t * stride + h2 - stride, t # the new length, the conv times

def get_pointer(v): # as float
    return v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))