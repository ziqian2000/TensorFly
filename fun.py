import numpy as np
from tensorfly.helper import *
from tensorfly.nodes import *

float32 = np.float32
float64 = np.float64

def gradients(output_node, node_list):
	"""Take gradient of output node with respect to each node in node_list.
	Parameters
	----------
	output_node: output node that we are taking derivative of.
	node_list: list of nodes that we are taking derivative wrt.
	Returns
	-------
	A list of gradient values, one for each node in node_list respectively.
	"""

	# a map from node to a list of gradient contributions from each output node
	node_to_output_grads_list = {}
	# Special note on initializing gradient of output_node as oneslike_op(output_node):
	# We are really taking a derivative of the scalar reduce_sum(output_node)
	# instead of the vector output_node. But this is the common case for loss function.
	node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
	# a map from node to the gradient of that node
	node_to_output_grad = {}
	# Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
	reverse_topo_order = reversed(find_topo_sort([output_node]))

	
	for node in reverse_topo_order:
		node_to_output_grad[node] = sum_node_list(node_to_output_grads_list[node])
		inputs_grad = node.op.gradient(node, node_to_output_grad[node])
		for i in range(len(node.inputs)):
			p = inputs_grad[i]
			if(node.inputs[i] in node_to_output_grads_list):
				node_to_output_grads_list[node.inputs[i]].append(p)
			else:
				node_to_output_grads_list[node.inputs[i]] = [p]

	# Collect results for gradients requested.
	grad_node_list = [node_to_output_grad[node] for node in node_list]
	return grad_node_list

class train:

	class Optimizer:
		def get_variables_need(self, loss):
			nodes_need = find_topo_sort([loss])
			variables_need = []
			for node in nodes_need:
				if isinstance(node.op, VariableOp):
					variables_need.append(node)
			return variables_need

	class GradientDescentOptimizer(Optimizer):
		def __init__(self, learning_rate, use_locking = False, name = 'GradientDescent'):
			assert use_locking == False # simplify
			self.learning_rate = learning_rate
			self.name = name

		def minimize(self, loss):
			variables_need = self.get_variables_need(loss)
			gradients_need = gradients(loss, variables_need)
			train_node = [assign(variables_need[i], variables_need[i] - gradients_need[i] * self.learning_rate) for i in range(len(variables_need))]
			return pack(train_node)

	class AdamOptimizer(Optimizer):
		def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08, use_locking = False, name = 'Adam'):
			assert use_locking == False # simplify
			self.learning_rate = learning_rate
			self.beta1 = beta1
			self.beta2 = beta2
			self.epsilon = epsilon
			self.name = name

		def minimize(self, loss):
			self.t = constant(0)
			self.assign_t = assign(self.t, self.t + 1)
			self.lrt = adam_calc_op(self.assign_t, learning_rate = self.learning_rate, beta1 = self.beta1, beta2 = self.beta2, epsilon = self.epsilon)

			variables_need = self.get_variables_need(loss)
			gradients_need = gradients(loss, variables_need)
			train_node = []
			for i, node in enumerate(variables_need):
				g = gradients_need[i]
				m = constant(0)
				assign_m = assign(m, (1 - self.beta1) * g + self.beta1 * m)
				v = constant(0)
				assign_v = assign(v, (1 - self.beta2) * g * g + self.beta2 * v)
				assign_node = assign(node, node - self.lrt * assign_m / (sqrt_op(assign_v) + self.epsilon))
				train_node.append(assign_node)
			return pack(train_node)


def random_normal(shape, mean = 0.0, stddev = 1.0, dtype = float32, seed = None, name = None):
	# np.random.seed(123)
	return np.random.normal(loc = mean, scale = stddev, size = shape).astype(dtype)
