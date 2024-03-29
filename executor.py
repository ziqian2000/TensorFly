import numpy as np
from tensorfly.helper import *

Variable_assign_node_list = []

class Executor:
	"""Executor computes values for a given subset of nodes in a computation graph.""" 
	def __init__(self, eval_node_list = []):
		"""
		Parameters
		----------
		eval_node_list: list of nodes whose values need to be computed.
		"""
		self.eval_node_list = eval_node_list

	def run(self, feed_dict = {}):
		"""Computes values of nodes in eval_node_list given computation graph.
		Parameters
		----------
		feed_dict: list of variable nodes whose values are supplied by user.
		Returns
		-------
		A list of values for nodes in eval_node_list. 
		"""

		node_to_val_map = dict(feed_dict)
		# Traverse graph in topological sort order and compute values for all nodes.
		topo_order = find_topo_sort(self.eval_node_list)

		# make sure the data type is np.ndarray
		for node in node_to_val_map:
			if not isinstance(node_to_val_map[node], np.ndarray):
				node_to_val_map[node] = np.array(node_to_val_map[node])

			# if node_to_val_map[node].dtype == np.float64:
			# 	node_to_val_map[node] = node_to_val_map[node].astype(np.float32)
			# if node_to_val_map[node].dtype == np.int64:
			# 	node_to_val_map[node] = node_to_val_map[node].astype(np.int32)

		for node in topo_order:
			if node in node_to_val_map:
				node_to_val_map[node] = np.array(node_to_val_map[node], dtype = node.const_attr[0])
			else:
				node_to_val_map[node] = node.op.compute(node, [node_to_val_map[p] for p in node.inputs])

				# if(isinstance(node_to_val_map[node], np.ndarray)) and node_to_val_map[node].dtype != np.float32:
				# 	print(node.op)
				# 	print("->" , node_to_val_map[node].dtype)
				# 	for i in node.inputs: 
				# 		if(isinstance(node_to_val_map[i], np.ndarray)):
				# 			print(node_to_val_map[i].dtype)
				# else:
				# 	print("?", node.op)

				# print(type(node.op))
				# if(isinstance(node_to_val_map[node], np.ndarray)):
				# 	print(np.max(np.abs(node_to_val_map[node])), np.min(np.abs(node_to_val_map[node])))

		# Collect node values.
		node_val_results = [node_to_val_map[node] for node in self.eval_node_list]

		return node_val_results

class Session:
	def run(self, fetches, feed_dict = {}, options = None, run_metadata = None):
		if isinstance(fetches, list):
			self.exe = Executor(fetches)
			return self.exe.run(feed_dict)
		else:
			self.exe = Executor([fetches])
			return self.exe.run(feed_dict)[0]

	def __enter__(self):
		return self

	def __exit__(self, val, type, trace):
		pass