<<<<<<< HEAD
from node import Node
from element import Element
from util import Counter
import numpy as np
=======
from .node import Node
>>>>>>> b54520e4681fb524d8b8e4e9df3ff1379b4c85f5

class Vector(Node):
	def __init__(self, val, der = 1):
		""" Returns a Vector variable with user defined value and derivative

<<<<<<< HEAD
		INPUTS
		=======
		val: float, compulsory
			Value of the Vector variable
		der: float, optional, default value is 1
			Derivative of the Vector variable/function of a variable
=======
class Vector(Node):
    def __init__(self):
        pass
>>>>>>> b54520e4681fb524d8b8e4e9df3ff1379b4c85f5

		RETURNS
		========
		Vector class instance

		NOTES
		=====
		PRE:
			- val and der have numeric type
			- two or fewer inputs
		POST:
			returns a Vector class instance with value = val and derivative = der

		EXAMPLES
		=========
		>>> Vector(2, 1)
		Vector variable with value 2
		"""
		self._val = np.array(val)
		self._jacobian = der * np.eye(len(val))

	def __getitem__(self, idx):
		return Element(self._val[idx], self._jacobian[idx], self)

	def __add__(self, other):
		""" Returens the sum of self and other

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> y = Vector(2,1)
		>>> x+y
		Vector variable with value 3
		"""
		try:
			value = self._val + other._val # If other is a constant, then there will be an attribute error
		except AttributeError:
			value = self._val + other
			try: 
				dict_self = self._dict # If self is a user defined variable, then there will be an attribute error
				new = Vector(value, self._jacobian)
				new._dict = dict_self # When self is a complex function and other is a constant, the derivatives of the sum is just the derivatives of self
				return new
			except AttributeError:
				derivative = Counter() # If self is a user defined variable, then we add a dictionary of derivatives of user defined variables to the result Vector variable
				derivative[self] = self._jacobian
				new = Vector(value, self._jacobian)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict 
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._jacobian
				try:
					dict_other = other._dict # If other is a user defined variable, then there will be an attribute error
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian  # If other is a user defined variable, then we initiate a Counter dictionary for it
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key] # Then the derivatives of result Vector variable are sums of derivatives of self and derivatives of other
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key] # If self and other are both complex functions, then the derivatives of result Vector variable are sums of derivatives of self and derivatives of other
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key]
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key]
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new


	def __radd__(self, other):
		""" Returens the sum of self and other

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> y = Vector(2,1)
		>>> x+y
		Vector variable with value 3
		"""
		return self + other

	def __sub__(self, other):
		""" Returens the difference of self and other

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> y = Vector(2,1)
		>>> x-y
		Vector variable with value -1
		"""
		try:
			value = self._val - other._val # If other is a constant, then there will be an attribute error
		except AttributeError:
			value = self._val - other
			try: 
				dict_self = self._dict # If self is a user defined variable, then there will be an attribute
				new = Vector(value, self._jacobian)
				new._dict = dict_self
				return new
			except AttributeError:
				derivative = Counter()
				derivative[self] = self._jacobian
				new = Vector(value, self._jacobian)
				new._dict = derivative # When self is a complex function and other is a constant, the derivatives of the result variable is just the derivatives of self
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter() # If self is a user defined variable, then we add a dictionary of derivatives of user defined variables to the result Vector variable
				dict_self[self] = self._jacobian
				try:
					dict_other = other._dict # If other is a user defined variable, then there will be an attribute error
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian # If other is a user defined variable, then we initiate a Counter dictionary for it
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key] # Then the derivatives of result Vector variable are differences of derivatives of self and derivatives of other
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key] # If self and other are both complex functions, then the derivatives of result Vector variable are differences of derivatives of self and derivatives of other
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key]
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key]
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new

	def __rsub__(self, other):
		""" Returens the difference of other and self

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> y = Vector(2,1)
		>>> x-y
		Vector variable with value -1
		"""
		return self.__neg__() + other

	def __mul__(self, other):
		""" Returens the product of other and self

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> y = Vector(2,1)
		>>> x * y
		Vector variable with value 2
		"""
		try:
			val_other = other._val # If other is a constant, then there will be an attribute error
			value = self._val * other._val
		except AttributeError:
			val_other = other
			value = self._val * other
			try:
				dict_self = self._dict # If self is a user defined variable, then there will be an attribute
				for key in dict_self.keys():
					dict_[key] = dict_self[key] * val_other
				new = Vector(value, self._jacobian)
				new._dict = dict_self # When self is a complex function and other is a constant, the derivatives of the result variable is just the derivatives of self
				return new
			except AttributeError:
				derivative = Counter() # If self is a user defined variable, then we add a dictionary of derivatives of user defined variables to the result Vector variable
				derivative[self] = self._jacobian * val_other
				new = Vector(value, self._jacobian)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._jacobian 
				try:
					dict_other = other._dict # If other is a user defined variable, then there will be an attribute error
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian # If other is a user defined variable, then we initiate a Counter dictionary for it
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val # Then the derivatives of result Vector variable are sum of products of derivatives and values
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val # If self and other are both complex functions,then the derivatives of result Vector variable are sum of products of derivatives and values
					new = Vector(value, self._jacobian)
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new

	def __rmul__(self, other):
		""" Returens the product of other and self

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> y = Vector(2,1)
		>>> x * y
		Vector variable with value 2
		"""
		return self * other

	def __truediv__(self, other):
		""" Returens the quotient of self and other

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> x / 2
		Vector variable with value 0.5
		"""
		try:
			val_other = other._val
			if(0 not in val_other):
				value = self._val / val_other
			else:
				print('Divisor could not be 0')
		except AttributeError:
			val_other = other
			if(0 not in val_other):
				value = self._val / val_other
			else:
				print('Divisor could not be 0')
			try:
				dict_self = self._dict
				for key in dict_self.keys():
					dict_self[key] = dict_self[key] / val_other
				new = Vector(value, self._jacobian)
				new._dict = dict_self
				return new
			except AttributeError:
				derivative = Counter()
				derivative[self] = self._jacobian / val_other
				new = Vector(value, self._jacobian)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._jacobian
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(0 not in val_other):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(0 not in val_other):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(0 not in val_other):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(0 not in val_other):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new

	def __rtruediv__(self, other):
		""" Returens the quotient of other and self

		INPUTS
		=======
		self: this Vector class instance, compulsory
		other: constant or Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> 2 / x
		Vector variable with value 2
		"""
		try:
			val_other = other._val
			if(self._val != 0):
				value = val_other / self._val
			else:
				print('Divisor could not be 0')
		except AttributeError:
			val_other = other
			if(self._val != 0):
				value = val_other / self._val
			else:
				print('Divisor could not be 0')
			try:
				dict_self = self._dict
				for key in dict_self.keys():
					dict_self[key] = - val_other * dict_self[key] / (self._val * self._val)
				new = Vector(value, self._jacobian)
				new._dict = dict_self
				return new
			except AttributeError:
				derivative = Counter()
				derivative[self] = - val_other * self._jacobian / (self._val * self._val)
				new = Vector(value, self._jacobian)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._jacobian
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(self._val != 0):
							derivative[key] = dict_other[key] / self._val - dict_self[key] * val_other / (self._val * self._val)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(self._val != 0):
							derivative[key] = dict_other[key] / self._val - dict_self[key] * val_other / (self._val * self._val)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._jacobian
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(self._val != 0):
							derivative[key] = dict_other[key] / self._val - dict_self[key] * val_other / (self._val * self._val)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(self._val != 0):
							derivative[key] = dict_other[key] / self._val - dict_self[key] * val_other / (self._val * self._val)
						else:
							print('Divisor could not be 0')
					new = Vector(value, self._jacobian)
					new._dict = derivative
					return new

	def __neg__(self):
		""" Returens the product of -1 and self

		INPUTS
		=======
		self: this Vector class instance, compulsory

		RETURNS
		========
		Vector class instance 

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> - x
		Vector variable with value -1
		"""
		value = - self._val
		derivative = - self._jacobian
		new = Vector(value, derivative)
		try:
			dict_self = self._dict
		except AttributeError:
			return new
		else:
			for key in dict_self.keys():
				dict_self[key] = - dict_self[key]
			new._dict = dict_self
			return new

	def getDerivative(self, x):
		""" Returens the derivative of function self of variable x

		INPUTS
		=======
		self: this Vector class instance, compulsory
		x: user defined variable, compulsory

		RETURNS
		========
		derivative of function self of user defined variable x: float

		NOTES
		=====
		PRE:
			- x must be user defined variable
		POST:
			- returns a float derivative

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> y = Vector(3,1)
		>>> f = 2 * x + x * y
		>>> f.getDerivative(x)
		5
		"""
		return self._dict[x]

	def __repr__(self):
		""" Returens a description about the Vector variable class instance

		INPUTS
		=======
		self: this Vector class instance, compulsory

		RETURNS
		========
		description about the Sclar variable class instance: string

		EXAMPLES
		=========
		>>> x = Vector(1,1)
		>>> print(x)
		'Vector variable with value 1'
		"""
		representation = 'Vector variable with value {}'.format(self._val)
		return representation
