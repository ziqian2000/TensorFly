import numpy as np
from tensorfly.connection import *
from tensorfly.executor import *
from tensorfly.helper import *

timer = 0

class Node(object):
	"""Node in a computation graph."""
	def __init__(self):
		"""Constructor, new node is indirectly created by Op object __call__ method.
			
			Instance variables
			------------------
			self.inputs: the list of input nodes.
			self.op: the associated op object, 
				e.g. add_op object if this node is created by adding two other nodes.
			self.const_attr: the add or multiply constant,
				e.g. self.const_attr=5 if this node is created by x+5.
			self.name: node name for debugging purposes.
		"""
		self.inputs = []
		self.op = None
		self.const_attr = None
		self.name = ""

	def __add__(self, other):
		"""Adding two nodes return a new node."""
		if isinstance(other, Node):
			new_node = add_op(self, other)
		else:
			# Add by a constant stores the constant in the new node's const_attr field.
			# 'other' argument is a constant
			new_node = add_byconst_op(self, other)
		return new_node

	def __mul__(self, other):
		
		if isinstance(other, Node):
			new_node = mul_op(self, other)
		else:
			new_node = mul_byconst_op(self, other)
		return new_node

	# Allow left-hand-side add and multiply.
	__radd__ = __add__
	__rmul__ = __mul__

	def __sub__(self, other):
		if isinstance(other, Node):
			new_node = sub_op(self, other)
		else:
			new_node = sub_op(self, constant(other))
		return new_node

	def __rsub__(self, other):
		if isinstance(other, Node):
			new_node = sub_op(other, self)
		else:
			new_node = sub_op(constant(other), self)
		return new_node

	def __truediv__(self, other):
		if not isinstance(other, Node):
			other = constant(other)
		return div_op(self, other)

	def __neg__(self):
		new_node = neg_op(self)
		return new_node

	def __str__(self):
		"""Allow print to display node name.""" 
		return self.name

	__repr__ = __str__

	def eval(self, feed_dict = {}):
		self.exe = Executor([self])
		return self.exe.run(feed_dict)[0]

	run = eval

class Op(object):
	"""Op represents operations performed on nodes."""
	def __call__(self):
		"""Create a new node and associate the op object with the node.
		
		Returns
		-------
		The new node object.
		"""
		new_node = Node()
		new_node.op = self
		return new_node

	def compute(self, node, input_vals):
		"""Given values of input nodes, compute the output value.
		Parameters
		----------
		node: node that performs the compute.
		input_vals: values of input nodes.
		Returns
		-------
		An output value of the node.
		"""
		raise NotImplementedError

	def gradient(self, node, output_grad):
		"""Given value of output gradient, compute gradient contributions to each input node.
		Parameters
		----------
		node: node that performs the gradient.
		output_grad: value of output gradient summed from children nodes' contributions
		Returns
		-------
		A list of gradient contributions to each input node respectively.
		"""
		raise NotImplementedError

class NegOp(Op):
	def __call__(self, node_A):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.name = "(-%s)" % (node_A.name)
		return new_node

	def compute(self, node, input_vals):
		return -input_vals[0]

	def gradient(self, node, output_grad):
		return [-output_grad]

class AddOp(Op):
	"""Op to element-wise add two nodes."""
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
		return new_node

	def compute(self, node, input_vals):
		"""Given values of two input nodes, return result of element-wise addition."""
		return input_vals[0] + input_vals[1]

	def gradient(self, node, output_grad):
		"""Given gradient of add node, return gradient contributions to each input."""
		return [reduce_shape(output_grad, node.inputs[0]), reduce_shape(output_grad, node.inputs[1])]

class AddByConstOp(Op):
	"""Op to element-wise add a nodes by a constant."""
	def __call__(self, node_A, const_val):
		new_node = Op.__call__(self)
		new_node.const_attr = const_val
		new_node.inputs = [node_A]
		new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
		return new_node

	def compute(self, node, input_vals):
		"""Given values of input node, return result of element-wise addition."""
		return input_vals[0] + node.const_attr

	def gradient(self, node, output_grad):
		"""Given gradient of add node, return gradient contribution to input."""
		return [output_grad]

class SubOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
		return new_node

	def compute(self, node, input_vals):
		return input_vals[0] - input_vals[1]

	def gradient(self, node, output_grad):
		return [output_grad, -output_grad]

class MulOp(Op):
	"""Op to element-wise multiply two nodes."""
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
		return new_node

	def compute(self, node, input_vals):
		"""Given values of two input nodes, return result of element-wise multiplication."""
		return input_vals[0] * input_vals[1]

	def gradient(self, node, output_grad):
		"""Given gradient of multiply node, return gradient contributions to each input."""
		return [output_grad * node.inputs[1], output_grad * node.inputs[0]]

class MulByConstOp(Op):
	"""Op to element-wise multiply a nodes by a constant."""
	def __call__(self, node_A, const_val):
		new_node = Op.__call__(self)
		new_node.const_attr = const_val
		new_node.inputs = [node_A]
		new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
		return new_node

	def compute(self, node, input_vals):
		"""Given values of input node, return result of element-wise multiplication."""
		
		return input_vals[0] * node.const_attr

	def gradient(self, node, output_grad):
		"""Given gradient of multiplication node, return gradient contribution to input."""
		return [node.const_attr * output_grad]

class MatMulOp(Op):
	"""Op to matrix multiply two nodes."""
	def __call__(self, node_A, node_B, trans_A = False, trans_B = False):
		"""Create a new node that is the result a matrix multiple of two input nodes.
		Parameters
		----------
		node_A: lhs of matrix multiply
		node_B: rhs of matrix multiply
		trans_A: whether to transpose node_A
		trans_B: whether to transpose node_B
		Returns
		-------
		Returns a node that is the result a matrix multiple of two input nodes.
		"""
		new_node = Op.__call__(self)
		new_node.matmul_attr_trans_A = trans_A
		new_node.matmul_attr_trans_B = trans_B
		new_node.inputs = [node_A, node_B]
		new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
		return new_node

	def compute(self, node, input_vals):
		a, b = input_vals
		a = a.astype(np.float32)
		b = b.astype(np.float32)
		na, ma = a.shape
		nb, mb = b.shape
		if node.matmul_attr_trans_A:
			if node.matmul_attr_trans_B: 	c = np.ndarray(shape = (ma, nb), dtype = np.float32)
			else:							c = np.ndarray(shape = (ma, mb), dtype = np.float32)
		else:
			if node.matmul_attr_trans_B:	c = np.ndarray(shape = (na, nb), dtype = np.float32)
			else:							c = np.ndarray(shape = (na, mb), dtype = np.float32)
		c_core.matmul_trans(get_pointer(a), get_pointer(b), get_pointer(c), na, ma, nb, mb, 
						int(node.matmul_attr_trans_A), int(node.matmul_attr_trans_B))
		return c

	def gradient(self, node, output_grad):
		"""Given gradient of multiply node, return gradient contributions to each input.
			
		Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
		"""
		return [matmul(output_grad, node.inputs[1], False, not node.matmul_attr_trans_B), 
				matmul(node.inputs[0], output_grad, not node.matmul_attr_trans_A, False)]

class DivOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
		return new_node

	def compute(self, node, input_vals):
		"""Given values of two input nodes, return result of element-wise multiplication."""
		return np.divide(input_vals[0], input_vals[1])

	def gradient(self, node, output_grad):
		return [output_grad / node.inputs[1], output_grad * (-node.inputs[0]/node.inputs[1]/node.inputs[1])]

class AssignOp(Op):
	def __call__(self, ref, value, validate_shape = None, use_locking = None, name = None):
		new_node = Op.__call__(self)
		if not isinstance(value, Node):
			value = constant(value)
		new_node.inputs = [value]
		new_node.const_attr = ref
		new_node.name = name
		return new_node

	def compute(self, node, input_vals):
		node.const_attr.const_attr = input_vals[0]
		return input_vals[0]

	def gradient(self, node, output_grad):
		raise NotImplementedError

class VariableOp(Op):
	"""mostly the same as PlaceholderOp"""
	def __call__(self, initilal_value = None, name = "Variable", dtype = None):
		new_node = Op.__call__(self)
		if initilal_value is not None:
			assign_node = assign(new_node, initilal_value)
			Variable_assign_node_list.append(assign_node)
		new_node.const_attr = None
		new_node.name = name
		return new_node

	def compute(self, node, input_vals):
		return node.const_attr

	def gradient(self, node, output_grad):
		return None

class ConstantOp(Op):
	def __call__(self, value, dtype = None, shape = None, name = 'Const'):
		new_node = Op.__call__(self)
		if not isinstance(value, np.ndarray):
			value = np.array(value)
		if(shape != None):
			value = np.ones(shape = shape) * value
		if(dtype != None):
			value = value.astype(dtype)
		new_node.const_attr = value
		new_node.name = name
		return new_node

	def compute(self, node, input_vals):
		return node.const_attr

	def gradient(self, node, output_grad):
		return None

class PlaceholderOp(Op):
	"""Op to feed value to a nodes."""
	def __call__(self, dtype, shape = None, name = "Placeholder"):
		"""Creates a variable node."""
		new_node = Op.__call__(self)
		new_node.const_attr = (dtype, shape)
		new_node.name = name
		return new_node

	def compute(self, node, input_vals):
		"""No compute function since node value is fed directly in Executor."""
		raise NotImplementedError

	def gradient(self, node, output_grad):
		"""No gradient function since node has no inputs."""
		return None

class ZerosLikeOp(Op):
	"""Op that represents a constant np.zeros_like."""
	def __call__(self, node_A):
		"""Creates a node that represents a np.zeros array of same shape as node_A."""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.name = "Zeroslike(%s)" % node_A.name
		return new_node

	def compute(self, node, input_vals):
		"""Returns zeros_like of the same shape as input."""
		return np.zeros(input_vals[0].shape)

	def gradient(self, node, output_grad):
		return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
	"""Op that represents a constant np.ones_like."""
	def __call__(self, node_A):
		"""Creates a node that represents a np.ones array of same shape as node_A."""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.name = "Oneslike(%s)" % node_A.name
		return new_node

	def compute(self, node, input_vals):
		"""Returns ones_like of the same shape as input."""
		return np.ones(input_vals[0].shape)

	def gradient(self, node, output_grad):
		return [zeroslike_op(node.inputs[0])]

class VariableInitOp(Op):
	def __call__(self):
		new_node = Op.__call__(self)
		new_node.name = "VariableInit"
		return new_node

	def compute(self, node, input_vals):
		exe = Executor(Variable_assign_node_list)
		exe.run()

	def gradient(self, node, output_grad):
		raise NotImplementedError

class ReduceSumOp(Op):
	def __call__(self, input_tensor, axis = None, keepdims = False, name = None, reduction_indices = None):
		new_node = Op.__call__(self)
		new_node.inputs = [input_tensor]
		if axis is None and reduction_indices is not None:
			axis = reduction_indices
		if isinstance(axis, list):
			axis = axis[0]
		new_node.const_attr = (axis, keepdims)
		new_node.name = name if name != None else "ReduceSum(%s)" % input_tensor.name
		return new_node

	def compute(self, node, input_vals):
		return np.sum(input_vals[0], axis = node.const_attr[0], keepdims = node.const_attr[1])

	def gradient(self, node, output_grad):
		return [adaptive_broadcast_to_op(output_grad, node.inputs[0], node.const_attr[0], node.const_attr[1])]

class ReduceMeanOp(Op):
	def __call__(self, input_tensor, axis = None, keepdims = False, name = None, reduction_indices = None):
		new_node = Op.__call__(self)
		new_node.inputs = [input_tensor]
		if axis is None and reduction_indices is not None:
			axis = reduction_indices
		if isinstance(axis, list):
			axis = axis[0]
		new_node.const_attr = (axis, keepdims)
		new_node.name = name if name != None else "ReduceMean(%s)" % input_tensor.name
		return new_node

	def compute(self, node, input_vals):
		return np.mean(input_vals[0], axis = node.const_attr[0], keepdims = node.const_attr[1])

	def gradient(self, node, output_grad):
		return [adaptive_broadcast_to_op(output_grad, node.inputs[0], node.const_attr[0], node.const_attr[1])
				/ reduce_sum(oneslike_op(node.inputs[0]), axis = node.const_attr[0], keepdims = True)]

class AdaptiveBroadcastToOp(Op):
	""" transform 'tensor' into new 'shape' by inserting an axis in position 'axis' and repeating elements on that axis """
	def __call__(self, tensor, tensor_input, axis, keepdims, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [tensor, tensor_input]
		new_node.const_attr = (axis, keepdims)
		new_node.name = name if name != None else "BroadcastTo(%s,axis=%s)" % (tensor.name, axis)
		return new_node

	def compute(self, node, input_vals):
		axis, keepdims = node.const_attr
		val = input_vals[0]
		if(keepdims):
			val = np.sum(val, axis)
		if axis is None:
			return np.ones(input_vals[1].shape) * val
		else:
			return np.broadcast_to(np.expand_dims(val, axis = axis), input_vals[1].shape)

	def gradient(self, node, output_grad):
		raise NotImplementedError


class ZerosOp(Op):
	def __call__(self, shape, dtype = np.float32, name = None):
		new_node = Op.__call__(self)
		new_node.const_attr = np.zeros(shape)
		new_node.name = name
		return new_node

	def compute(self, node, input_vals):
		return node.const_attr

	def gradient(self, node ,output_grad):
		return None

class SoftmaxJointOp(Op):
	"""available gradient"""
	def __call__(self, logits, axis = -1, name = None):
		exp_node = exp(logits)
		return exp_node / reduce_sum(exp_node, axis = axis, keepdims = True)

	def compute(self, node, input_vals):
		raise NotImplementedError

	def gradient(self, node ,output_grad):
		raise NotImplementedError

class SoftmaxCalcOp(Op):
	"""unavailable gradient"""
	def __call__(self, x, axis = -1, name = None):
		new_node = Op.__call__(self)
		new_node.const_attr = axis
		new_node.inputs = [x]
		new_node.name = name if name != None else "SoftmaxCalc(%s)" % x.name
		return new_node

	def compute(self, node, input_vals):
		return SoftmaxCalcFunc(input_vals[0])

	def gradient(self, node ,output_grad):
		raise NotImplementedError


class ExpOp(Op):
	def __call__(self, x, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [x]
		new_node.name = name if name != None else "Exp(%s)" % x.name
		return new_node

	def compute(self, node, input_vals):
		return np.exp(input_vals[0])

	def gradient(self, node, output_grad):
		return [node * output_grad]

class LogOp(Op):
	def __call__(self, x, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [x]
		new_node.name = name if name != None else "Log(%s)" % x.name
		return new_node

	def compute(self, node, input_vals):
		return np.log(input_vals[0])

	def gradient(self, node, output_grad):
		return [output_grad / node.inputs[0]]

class EqualOp(Op):
	def __call__(self, x, y, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [x, y]
		new_node.name = name if name != None else "Equal(%s,%s)" % (x.name, y.name)
		return new_node

	def compute(self, node, input_vals):
		return np.equal(input_vals[0], input_vals[1])

	def gradient(self, node, output_grad):
		raise NotImplementedError

class ArgMaxOp(Op):
	def __call__(self, input, axis = None, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [input]
		new_node.const_attr = axis
		new_node.name = name if name != None else "ArgMax(%s,axis=%s)" % (input.name, axis)
		return new_node

	def compute(self, node, input_vals):
		return np.argmax(input_vals[0], axis = node.const_attr)

	def gradient(self, node, output_grad):
		raise NotImplementedError

class CastOp():
	def __call__(self, x, dtype, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [x]
		new_node.const_attr = dtype
		new_node.name = name if name != None else "Cast(%s,dtype=%s)" % (x.name, dtype)
		return new_node

	def compute(self, node, input_vals):
		return input_vals[0].astype(node.const_attr)

	def gradient(self, node, output_grad):
		raise NotImplementedError

class PackOp(Op):
	def __call__(self, nodes_list):
		new_node = Op.__call__(self)
		new_node.inputs = nodes_list
		return new_node

	def compute(self, node, input_vals):
		return None

	def gradient(self, node, output_grad):
		raise NotImplementedError

class ReduceShapeOp(Op):
	'''reduce the shape of tensor to target by np.sum(axis = 0)
	   this op is used in gradients function of AddOp since the add operator is right(high dim)-aligned'''
	def __call__(self, tensor, target):
		new_node = Op.__call__(self)
		new_node.inputs = [tensor, target]
		return new_node

	def compute(self, node, input_vals):
		tensor = input_vals[0]
		shape = input_vals[1].shape
		while len(tensor.shape) > len(shape):
			tensor = np.sum(tensor, axis = 0)
		return tensor

	def gradient(self, node, output_grad):
		raise NotImplementedError

class ReluOp(Op):
	def __call__(self, tensor):
		new_node = Op.__call__(self)
		new_node.inputs = [tensor]
		return new_node

	def compute(self, node, input_vals):
		input = input_vals[0].astype(np.float32)
		output = np.ndarray(shape = input.shape, dtype = np.float32)
		c_core.relu(get_pointer(input), get_pointer(output), input.size)
		return output

	def gradient(self, node, output_grad):
		return [relu_gradient(node.inputs[0], output_grad)]

class ReluGradientOp():
	def __call__(self, tensor, grad):
		new_node = Op.__call__(self)
		new_node.inputs = [tensor, grad]
		return new_node

	def compute(self, node, input_vals):
		input = input_vals[0].astype(np.float32)
		grad = input_vals[1].astype(np.float32)
		output = np.ndarray(shape = input.shape, dtype = np.float32)
		c_core.sgn_zero_or_posi(get_pointer(input), get_pointer(grad), get_pointer(output), input.size)
		return output

	def gradient(self, node, output_grad):
		raise NotImplementedError

class SoftmaxCrossEntropyWithLogitsOp(Op):
	def __call__(self, labels, logits):
		new_node = Op.__call__(self)
		new_node.inputs = [logits, labels]
		return new_node

	def compute(self, node, input_vals):
		tmp = SoftmaxCalcFunc(input_vals[0])
		log_node = np.log(tmp)
		return -np.sum(log_node * input_vals[1], axis = -1, keepdims = True)

	def gradient(self, node, output_grad):
		"""as the second gradient is useless so I just return zeros to make it faster"""
		return [output_grad * (softmax_calc_op(node.inputs[0]) - node.inputs[1]), zeroslike_op(node.inputs[1])]

class SqrtOp(Op):
	def __call__(self, x):
		new_node = Op.__call__(self)
		new_node.inputs = [x]
		return new_node

	def compute(self, node, input_vals):
		return np.sqrt(input_vals[0])

	def gradient(self, node, output_grad):
		raise NotImplementedError

class AdamCalcOp(Op):
	def __call__(self, t, learning_rate, beta1, beta2, epsilon):
		new_node = Op.__call__(self)
		new_node.inputs = [t]
		new_node.const_attr = (learning_rate, beta1, beta2, epsilon)
		return new_node

	def compute(self, node, input_vals):
		t = input_vals[0]
		learning_rate, beta1, beta2, epsilon = node.const_attr
		lrt = learning_rate * np.sqrt(1 - np.power(beta2, t)) / (1 - np.power(beta1, t))
		return lrt

	def gradient(self, node, output_grad):
		raise NotImplementedError

class Conv2dOp(Op):
	def __call__(self, input, filter, strides, padding):
		new_node = Op.__call__(self)
		new_node.inputs = [input, filter]
		new_node.const_attr = (strides, padding)
		return new_node

	def compute(self, node, input_vals):
		input, filter = input_vals
		strides, padding = node.const_attr
		return Conv2dFunc(input = input_vals[0], filter = input_vals[1], strides = node.const_attr[0], padding = node.const_attr[1], need_to_rotate = False)

	def gradient(self, node, output_grad):
		return [conv2d_grad_1_op(output_grad, node.inputs[1]), conv2d_grad_2_op(output_grad, node.inputs[0], node.inputs[1])]

class Conv2dGrad1Op(Op):
	""" assuming strides == [1,1,1,1] and padding == 'SAME' """
	def __call__(self, output_grad, filter):
		new_node = Op.__call__(self)
		new_node.inputs = [output_grad, filter]
		return new_node

	def compute(self, node, input_vals):
		"""maybe it can return zeros since it can not be changed"""
		return Conv2dFunc(input = input_vals[0], filter = input_vals[1], strides = [1,1,1,1], padding = 'SAME', need_to_rotate = True)

	def gradient(self, node, output_grad):
		raise NotImplementedError

class Conv2dGrad2Op(Op):
	""" assuming strides == [1,1,1,1] and padding == 'SAME' """
	def __call__(self, output_grad, input, filter):
		new_node = Op.__call__(self)
		new_node.inputs = [output_grad, input, filter]
		return new_node

	def compute(self, node, input_vals):
		output_grad, input, filter = input_vals

		n_h, o_h = calc_new_len(h1 = input.shape[1], h2 = filter.shape[0], stride = 1)
		n_w, o_w = calc_new_len(h1 = input.shape[2], h2 = filter.shape[1], stride = 1)
		input = input.astype(np.float32)
		output_grad = output_grad.astype(np.float32)
		output = np.zeros_like(filter, dtype = np.float32) # the tensor used for result
		c_core.conv2d_grad(get_pointer(input), 		input.shape[0], 		input.shape[1], 		input.shape[2],
						  get_pointer(output_grad), output_grad.shape[1], 	output_grad.shape[2],
						  get_pointer(output),		output.shape[0], 		output.shape[1],		output.shape[2],		output.shape[3],
						  (n_h - input.shape[1]) // 2, 						(n_w - input.shape[2]) // 2)

		return output


	def gradient(self, node, output_grad):
		raise NotImplementedError

class MaxPoolOp(Op):
	def __call__(self, value, ksize, strides, padding):
		new_node = Op.__call__(self)
		new_node.inputs = [value]
		new_node.const_attr = (ksize, strides, padding)
		new_node.id = ++timer
		return new_node

	def compute(self, node, input_vals):
		input = input_vals[0]
		ksize, strides, padding = node.const_attr
		n_h, o_h = calc_new_len(h1 = input.shape[1], h2 = ksize[1], stride = strides[1])
		n_w, o_w = calc_new_len(h1 = input.shape[2], h2 = ksize[2], stride = strides[2])
		input = input.astype(np.float32)
		output = np.zeros([input.shape[0], o_h, o_w, input.shape[3]], dtype = np.float32) # the tensor used for result
		c_core.maxpool( get_pointer(input),		input.shape[1],		input.shape[2],
						get_pointer(output),	output.shape[0],	output.shape[1],	output.shape[2],	output.shape[3],
						strides[1],			strides[2],
						(n_h - input.shape[1]) // 2, 				(n_w - input.shape[2]) // 2,			node.id)
		return output

	def gradient(self, node, output_grad):
		return [max_pool_grad_op(node.inputs[0], node.const_attr[0], node.const_attr[1], output_grad, node.id)]

class MaxPoolGradOp(Op):
	def __call__(self, input, ksize, strides, output_grad, id):
		new_node = Op.__call__(self)
		new_node.inputs = [input, output_grad]
		new_node.const_attr = (ksize, strides, id)
		return new_node

	def compute(self, node, input_vals):
		input, output_grad = input_vals
		ksize, strides, id = node.const_attr
		n_h, o_h = calc_new_len(h1 = input.shape[1], h2 = ksize[1], stride = strides[1])
		n_w, o_w = calc_new_len(h1 = input.shape[2], h2 = ksize[2], stride = strides[2])
		output_grad = output_grad.astype(np.float32)
		output = np.zeros_like(input, dtype = np.float32)
		c_core.maxpool_grad(	get_pointer(output_grad), 	output_grad.shape[1], 	output_grad.shape[2],
								get_pointer(output),		output.shape[0], 		output.shape[1],		output.shape[2],		output.shape[3],
								strides[1],					strides[2],
								(n_h - input.shape[1]) // 2, 				(n_w - input.shape[2]) // 2,	id)
		return output


	def gradient(self, node, output_grad):
		raise NotImplementedError

class ReshapeOp(Op):
	def __call__(self, tensor, shape):
		new_node = Op.__call__(self)
		new_node.inputs = [tensor]
		new_node.const_attr = shape
		return new_node

	def compute(self, node, input_vals):
		return np.reshape(input_vals[0], node.const_attr)

	def gradient(self, node, output_grad):
		return [reshape_to_tensor_op(output_grad, node.inputs[0])]

class ReshapeToTensorOp(Op):
	"""tensor1 -> tensor2"""
	def __call__(self, tensor1, tensor2):
		new_node = Op.__call__(self)
		new_node.inputs = [tensor1, tensor2]
		return new_node

	def compute(self, node, input_vals):
		return np.reshape(input_vals[0], input_vals[1].shape)

	def gradient(self, node, output_grad):
		raise NotImplementedError

class DropOutMatOp(Op):
	def __call__(self, keep_prob, x):
		new_node = Op.__call__(self)
		new_node.inputs = [keep_prob, x]
		return new_node

	def compute(self, node, input_vals):
		return (np.random.uniform(size = input_vals[1].shape) < input_vals[0]).astype(np.int)

	def gradient(self, node, output_grad):
		return [zeroslike_op(node.inputs[0]), zeroslike_op(node.inputs[1])]


# Create global singletons of operators.
adam_calc_op = AdamCalcOp()
adaptive_broadcast_to_op = AdaptiveBroadcastToOp()
add_byconst_op = AddByConstOp()
add_op = AddOp()
argmax = ArgMaxOp()
assign = AssignOp()
cast = CastOp()
constant = ConstantOp()
conv2d_grad_1_op = Conv2dGrad1Op()
conv2d_grad_2_op = Conv2dGrad2Op()
div_op = DivOp()
dropout_mat_op = DropOutMatOp()
equal = EqualOp()
exp = ExpOp()
global_variables_initializer = VariableInitOp()
log = LogOp()
matmul = MatMulOp()
max_pool_grad_op = MaxPoolGradOp()
mul_byconst_op = MulByConstOp()
mul_op = MulOp()
neg_op = NegOp()
oneslike_op = OnesLikeOp()
pack = PackOp()
placeholder = PlaceholderOp()
reduce_mean = ReduceMeanOp()
reduce_shape = ReduceShapeOp()
reduce_sum = ReduceSumOp()
relu_gradient = ReluGradientOp()
reshape = ReshapeOp()
reshape_to_tensor_op = ReshapeToTensorOp()
softmax_calc_op = SoftmaxCalcOp() # unavailable gradient
sqrt_op = SqrtOp()
sub_op = SubOp()
Variable = VariableOp()
zeros = ZerosOp()
zeroslike_op = ZerosLikeOp()

class nn:
	conv2d = Conv2dOp()
	relu = ReluOp()
	softmax = SoftmaxJointOp() # available gradient
	softmax_cross_entropy_with_logits = SoftmaxCrossEntropyWithLogitsOp()
	max_pool = MaxPoolOp()
	def dropout(x, keep_prob): return x * dropout_mat_op(keep_prob, x) / keep_prob


def SoftmaxCalcFunc(tensor):
	exp_tensor = np.exp(tensor - np.max(tensor, axis = -1, keepdims = True))
	softmax_tensor = exp_tensor / np.sum(exp_tensor, axis = -1, keepdims = True)
	return softmax_tensor

def Conv2dFunc(input, filter, strides, padding, need_to_rotate = False):
	n_h, o_h = calc_new_len(h1 = input.shape[1], h2 = filter.shape[0], stride = strides[1])
	n_w, o_w = calc_new_len(h1 = input.shape[2], h2 = filter.shape[1], stride = strides[2])
	input = input.astype(np.float32)
	filter = filter.astype(np.float32)
	if(need_to_rotate):
		output = np.ndarray(shape = [input.shape[0], o_h, o_w, filter.shape[2]], dtype = np.float32) # the tensor used for result (1)
	else:
		output = np.ndarray(shape = [input.shape[0], o_h, o_w, filter.shape[3]], dtype = np.float32) # the tensor used for result (2)
	c_core.conv2d(	get_pointer(input), 	input.shape[0], 	input.shape[1], 	input.shape[2],
					get_pointer(filter), 	filter.shape[0], 	filter.shape[1], 	filter.shape[2], 	filter.shape[3],
					get_pointer(output),	output.shape[1],	output.shape[2],
					(n_h - input.shape[1]) // 2, 			(n_w - input.shape[2]) // 2,	int(need_to_rotate))
	return output