from node import Node
from util import Counter

class Scalar(Node):
	def __init__(self, val, der = 1):
		""" Returns a scalar variable with user defined value and derivative

		INPUTS
		=======
		val: float, compulsory
			Value of the scalar variable
		der: float, optional, default value is 1
			Derivative of the scalar variable/function of a variable

		RETURNS
		========
		Scalar class instance

		NOTES
		=====
		PRE:
			- val and der have numeric type
			- two or fewer inputs
		POST:
			returns a Scalar class instance with value = val and derivative = der

		EXAMPLES
		=========
		>>> Scalar(2, 1)
		Scalar variable with value 2
		"""
		self._val = val
		self._der = der

	def __add__(self, other):
		""" Returens the sum of self and other

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> y = Scalar(2,1)
		>>> x+y
		Scalar variable with value 3
		"""
		try:
			value = self._val + other._val # If other is a constant, then there will be an attribute error
		except AttributeError:
			value = self._val + other
			try: 
				dict_self = self._dict # If self is a user defined variable, then there will be an attribute error
				new = Scalar(value, self._der)
				new._dict = dict_self # When self is a complex function and other is a constant, the derivatives of the sum is just the derivatives of self
				return new
			except AttributeError:
				derivative = Counter() # If self is a user defined variable, then we add a dictionary of derivatives of user defined variables to the result Scalar variable
				derivative[self] = self._der
				new = Scalar(value, self._der)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict 
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._der
				try:
					dict_other = other._dict # If other is a user defined variable, then there will be an attribute error
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der  # If other is a user defined variable, then we initiate a Counter dictionary for it
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key] # Then the derivatives of result Scalar variable are sums of derivatives of self and derivatives of other
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key] # If self and other are both complex functions, then the derivatives of result Scalar variable are sums of derivatives of self and derivatives of other
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key]
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] + dict_other[key]
					new = Scalar(value, self._der)
					new._dict = derivative
					return new


	def __radd__(self, other):
		""" Returens the sum of self and other

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> y = Scalar(2,1)
		>>> x+y
		Scalar variable with value 3
		"""
		return self + other

	def __sub__(self, other):
		""" Returens the difference of self and other

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> y = Scalar(2,1)
		>>> x-y
		Scalar variable with value -1
		"""
		try:
			value = self._val - other._val # If other is a constant, then there will be an attribute error
		except AttributeError:
			value = self._val - other
			try: 
				dict_self = self._dict # If self is a user defined variable, then there will be an attribute
				new = Scalar(value, self._der)
				new._dict = dict_self
				return new
			except AttributeError:
				derivative = Counter()
				derivative[self] = self._der
				new = Scalar(value, self._der)
				new._dict = derivative # When self is a complex function and other is a constant, the derivatives of the result variable is just the derivatives of self
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter() # If self is a user defined variable, then we add a dictionary of derivatives of user defined variables to the result Scalar variable
				dict_self[self] = self._der
				try:
					dict_other = other._dict # If other is a user defined variable, then there will be an attribute error
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der # If other is a user defined variable, then we initiate a Counter dictionary for it
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key] # Then the derivatives of result Scalar variable are differences of derivatives of self and derivatives of other
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key] # If self and other are both complex functions, then the derivatives of result Scalar variable are differences of derivatives of self and derivatives of other
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key]
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] - dict_other[key]
					new = Scalar(value, self._der)
					new._dict = derivative
					return new

	def __rsub__(self, other):
		""" Returens the difference of other and self

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> y = Scalar(2,1)
		>>> x-y
		Scalar variable with value -1
		"""
		return self.__neg__() + other

	def __mul__(self, other):
		""" Returens the product of other and self

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> y = Scalar(2,1)
		>>> x * y
		Scalar variable with value 2
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
				new = Scalar(value, self._der)
				new._dict = dict_self # When self is a complex function and other is a constant, the derivatives of the result variable is just the derivatives of self
				return new
			except AttributeError:
				derivative = Counter() # If self is a user defined variable, then we add a dictionary of derivatives of user defined variables to the result Scalar variable
				derivative[self] = self._der * val_other
				new = Scalar(value, self._der)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._der 
				try:
					dict_other = other._dict # If other is a user defined variable, then there will be an attribute error
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der # If other is a user defined variable, then we initiate a Counter dictionary for it
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val # Then the derivatives of result Scalar variable are sum of products of derivatives and values
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val # If self and other are both complex functions,then the derivatives of result Scalar variable are sum of products of derivatives and values
					new = Scalar(value, self._der)
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_.keys()) + list(dict_other.keys())
					for key in lst:
						derivative[key] = dict_self[key] * val_other + dict_other[key] * self._val
					new = Scalar(value, self._der)
					new._dict = derivative
					return new

	def __rmul__(self, other):
		""" Returens the product of other and self

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> y = Scalar(2,1)
		>>> x * y
		Scalar variable with value 2
		"""
		return self * other

	def __truediv__(self, other):
		""" Returens the quotient of self and other

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> x / 2
		Scalar variable with value 0.5
		"""
		try:
			val_other = other._val
			if(val_other != 0):
				value = self._val / val_other
			else:
				print('Divisor could not be 0')
		except AttributeError:
			val_other = other
			if(val_other != 0):
				value = self._val / val_other
			else:
				print('Divisor could not be 0')
			try:
				dict_self = self._dict
				for key in dict_self.keys():
					dict_self[key] = dict_self[key] / val_other
				new = Scalar(value, self._der)
				new._dict = dict_self
				return new
			except AttributeError:
				derivative = Counter()
				derivative[self] = self._der / val_other
				new = Scalar(value, self._der)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._der
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(val_other != 0):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(val_other != 0):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(val_other != 0):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
				else:
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(val_other != 0):
							derivative[key] = dict_self[key] / val_other - dict_other[key] * self._val / (val_other * val_other)
						else:
							print('Divisor could not be 0')
					new = Scalar(value, self._der)
					new._dict = derivative
					return new

	def __rtruediv__(self, other):
		""" Returens the quotient of other and self

		INPUTS
		=======
		self: this Scalar class instance, compulsory
		other: constant or Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> 2 / x
		Scalar variable with value 2
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
				new = Scalar(value, self._der)
				new._dict = dict_self
				return new
			except AttributeError:
				derivative = Counter()
				derivative[self] = - val_other * self._der / (self._val * self._val)
				new = Scalar(value, self._der)
				new._dict = derivative
				return new
		else:
			try:
				dict_self = self._dict
			except AttributeError:
				dict_self = Counter()
				dict_self[self] = self._der
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(self._val != 0):
							derivative[key] = dict_other[key] / self._val - dict_self[key] * val_other / (self._val * self._val)
						else:
							print('Divisor could not be 0')
					new = Scalar(value, self._der)
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
					new = Scalar(value, self._der)
					new._dict = derivative
					return new
			else:
				try:
					dict_other = other._dict
				except AttributeError:
					dict_other = Counter()
					dict_other[other] = other._der
					derivative = Counter()
					lst = list(dict_self.keys()) + list(dict_other.keys())
					for key in lst:
						if(self._val != 0):
							derivative[key] = dict_other[key] / self._val - dict_self[key] * val_other / (self._val * self._val)
						else:
							print('Divisor could not be 0')
					new = Scalar(value, self._der)
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
					new = Scalar(value, self._der)
					new._dict = derivative
					return new

	def __neg__(self):
		""" Returens the product of -1 and self

		INPUTS
		=======
		self: this Scalar class instance, compulsory

		RETURNS
		========
		Scalar class instance 

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> - x
		Scalar variable with value -1
		"""
		value = - self._val
		derivative = - self._der
		new = Scalar(value, derivative)
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
		self: this Scalar class instance, compulsory
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
		>>> x = Scalar(1,1)
		>>> y = Scalar(3,1)
		>>> f = 2 * x + x * y
		>>> f.getDerivative(x)
		5
		"""
		return self._dict[x]

	def __repr__(self):
		""" Returens a description about the Scalar variable class instance

		INPUTS
		=======
		self: this Scalar class instance, compulsory

		RETURNS
		========
		description about the Sclar variable class instance: string

		EXAMPLES
		=========
		>>> x = Scalar(1,1)
		>>> print(x)
		'Scalar variable with value 1'
		"""
		representation = 'Scalar variable with value {}'.format(self._val)
		return representation
