import numpy as np
from tensorfly.executor import *

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
		assert isinstance(other, Node)
		return div_op(self, other)

	def __neg__(self):
		new_node = neg_op(self)
		return new_node

	def __str__(self):
		"""Allow print to display node name.""" 
		return self.name

	__repr__ = __str__

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
		assert len(input_vals) == 2
		return input_vals[0] + input_vals[1]

	def gradient(self, node, output_grad):
		"""Given gradient of add node, return gradient contributions to each input."""
		return [output_grad, output_grad]

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
		assert len(input_vals) == 1
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
		
		assert len(input_vals) == 2
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
		
		assert len(input_vals) == 1
		return input_vals[0] * node.const_attr

	def gradient(self, node, output_grad):
		"""Given gradient of multiplication node, return gradient contribution to input."""
		return [node.const_attr * output_grad]

class MatMulOp(Op):
	"""Op to matrix multiply two nodes."""
	def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
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
		"""Given values of input nodes, return result of matrix multiplication."""
		
		assert len(input_vals) == 2
		m0 = input_vals[0] if node.matmul_attr_trans_A == False else np.transpose(input_vals[0])
		m1 = input_vals[1] if node.matmul_attr_trans_B == False else np.transpose(input_vals[1])
		return np.matmul(m0, m1)

	def gradient(self, node, output_grad):
		"""Given gradient of multiply node, return gradient contributions to each input.
			
		Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
		"""
		return [matmul(output_grad, node.inputs[1], False, True), matmul(node.inputs[0], output_grad, True, False)]

class DivOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
		return new_node

	def compute(self, node, input_vals):
		"""Given values of two input nodes, return result of element-wise multiplication."""
		assert len(input_vals) == 2
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

	def gradient(self, node, output_grad):
		raise NotImplementedError

class VariableOp(Op):
	"""mostly the same as PlaceholderOp"""
	def __call__(self, initilal_value = None, name = "Variable", dtype = None):
		new_node = Op.__call__(self)
		if(initilal_value != None):
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
		assert False, "placeholder values provided by feed_dict"

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
			assert(len(axis) == 1) # simplify
			axis = axis[0]
		new_node.const_attr = (axis, keepdims)
		new_node.name = name if name != None else "ReduceSum(%s)" % input_tensor.name
		return new_node

	def compute(self, node, input_vals):
		return np.sum(input_vals[0], axis = node.const_attr[0], keepdims = node.const_attr[1])

	def gradient(self, node, output_grad):
		return [adaptive_broadcast_to_op(output_grad, node.const_attr[0], node.const_attr[1])]

class ReduceMeanOp(Op):
	def __call__(self, input_tensor, axis = None, keepdims = False, name = None, reduction_indices = None):
		new_node = Op.__call__(self)
		new_node.inputs = [input_tensor]
		if axis is None and reduction_indices is not None:
			axis = reduction_indices
		if isinstance(axis, list):
			assert(len(axis) == 1) # simplify
			axis = axis[0]
		new_node.const_attr = (axis, keepdims)
		new_node.name = name if name != None else "ReduceMean(%s)" % input_tensor.name
		return new_node

	def compute(self, node, input_vals):
		return np.mean(input_vals[0], axis = node.const_attr[0], keepdims = node.const_attr[1])

	def gradient(self, node, output_grad):
		return [adaptive_broadcast_to_op(output_grad, node.const_attr[0], node.const_attr[1])
				/ reduce_sum(oneslike_op(node.inputs[0]), axis = node.const_attr[0], keepdims = True)]

class AdaptiveBroadcastToOp(Op):
	""" transform 'tensor' into new 'shape' by inserting an axis in position 'axis' and repeating elements on that axis """
	def __call__(self, tensor, axis, keepdims, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [tensor]
		new_node.const_attr = (axis, keepdims)
		new_node.name = name if name != None else "BroadcastTo(%s,axis=%s)" % (tensor.name, axis)
		return new_node

	def compute(self, node, input_vals):
		axis, keepdims = node.const_attr
		val = input_vals[0]
		if(keepdims):
			val = np.sum(val, axis)
		if axis is None:
			return np.ones()
		else:
			ex_val = np.expand_dims(val, axis = axis)
			for i in range(np.shape(input_vals[0])[axis] - 1):
				ex_val = np.insert(ex_val, 0, val, axis = axis)
			return ex_val

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

class SoftmaxOp(Op):
	def __call__(self, logits, axis = -1, name = None):
		exp_node = exp(logits)
		return exp_node / reduce_sum(exp_node, axis = axis, keepdims = True)

	def compute(self, node, input_vals):
		raise NotImplementedError

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


# Create global singletons of operators.
add_byconst_op = AddByConstOp()
add_op = AddOp()
assign = AssignOp()
constant = ConstantOp()
global_variables_initializer = VariableInitOp()
log = LogOp()
matmul = MatMulOp()
mul_byconst_op = MulByConstOp()
mul_op = MulOp()
neg_op = NegOp()
oneslike_op = OnesLikeOp()
placeholder = PlaceholderOp()
reduce_mean = ReduceMeanOp()
reduce_sum = ReduceSumOp()
sub_op = SubOp()
Variable = VariableOp()
zeros = ZerosOp()
zeroslike_op = ZerosLikeOp()
adaptive_broadcast_to_op = AdaptiveBroadcastToOp()
div_op = DivOp()
exp = ExpOp()

class nn:
	softmax = SoftmaxOp()