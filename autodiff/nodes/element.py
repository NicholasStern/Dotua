class Element():
	def __init__(self, val, der, vector):
		self._val = val
		self._der = der
		self._vector = vector

	def __add__(self, other):
		try:
			value = self._val + other._val
			derivative = self._der + other._der
		except AttributeError:
			value = self._val + other
			derivative = self._der
			return Element(value, derivative, self._vector)
		else:
			if self._vector == other._vector:
				return Element(value, derivative, self._vector)
			else:
				print('Elements from different vectors are not allowed to go together')
				raise TypeError

	def __radd__(self, other):
		return self + other

	def __neg__(self):
		return Element(- self._val, - self._der, self._vector)

	def __sub__(self, other):
		try:
			value  = self._val - other._val
		except AttributeError:
			return Element(self._val - other, self._der, self._vector)
		else:
			if self._vector == other._vector:
				return self + other.__neg__()
			else:
				print('Elements from different vectors are not allowed to go together')
				raise TypeError


	def __rsub__(self, other):
		return self.__neg__() + other

	def __mul__(self, other):
		try:
			value = self._val * other._val
			derivative = self._der * other._val + self._val * other._der
		except AttributeError:
			value = self._val * other
			derivative = self._der * other
			return Element(value, derivative, self._vector)
		else:
			if self._vector == other._vector:
				return Element(value, derivative, self._vector)
			else:
				print('Elements from different vectors are not allowed to go together')
				raise TypeError

	def __rmul__(self, other):
		return self * other

	def __truediv__(self, other):
		try:
			val_other = other._val
		except AttributeError:
			val_other = other
			if(val_other != 0):
				value = self._val / val_other
				return Element(value, self._der / val_other, self._vector)
			else:
				raise ZeroDevisionError
				print("Divisor could not be 0")
		else:
			if self._vector == other._vector:
				val_other = other._val
				if(val_other != 0):
					value = self._val / val_other
					return Element(value, self._der / val_other - other._der * self._val / (val_other * val_other), self._vector)
				else:
					raise ZeroDevisionError
					print("Divisor could not be 0")
			else:
				print('Elements from different vectors are not allowed to go together')
				raise TypeError				

	def __rtruediv__(self, other):
		try:
			val_other = other._val 
		except AttributeError:
			val_other = other
			if(self._val != 0):
				value = val_other / self._val
				return Element(value, - val_other * self._der / (self._val * self._val), self._vector)
			else:
				raise ZeroDevisionError
				print("Divisor could not be 0")
		else:
			if self._vector == other._vector:
				if(self._val != 0):
					value = val_other / self._val
					return Element(value, other._der / self._val - val_other * self._der / (self._val * self._val), self._vector)
				else:
					raise ZeroDevisionError
					print("Divisor could not be 0")
			else:
				print("Elements from different vectors are not allowed to go together")
				raise TypeError		
	def __repr__(self):
		representation = 'Vector Element with value {} and derivative {}'.format(self._val, self._der)
		return representation

	def eval(self):
		return (self._val, self._der)