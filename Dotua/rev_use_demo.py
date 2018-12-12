import numpy as np

class Node():
    def __init__(self, val):
        self._val = val
        self._grad_val = 0
        self._roots = {}  # keys are input nodes, vals are lists intemediary nodes

    def init_roots(self):
        self._roots[self] = [(None, 1)]

    @staticmethod
    def sin(x):
        try:
            new_child = Node(np.sin(x._val))

            for input_var in x._roots.keys():
                try:
                    new_child._roots[input_var] += [(x, np.cos(x._val))]
                except KeyError:
                    new_child._roots[input_var] = [(x, np.cos(x._val))]
            return new_child
        except AttributeError:
            return np.sin(x)


    def gradient(self, input_var):
        for child, val in self._roots[input_var]:
            if child is not None:
                child._grad_val += self._grad_val * val
                child.gradient(input_var)



    # def __add__(self, other):
    #     new_parent = Node(self._val + other._val)
    #     try:
    #         new_parent.children.append((self, 1))
    #         new_parent.children.append((other, 1))
    #     except AttributeError:
    #         new_parent.children.append((self, 1))
    #     return new_parent

    def __mul__(self, other):
        try:
            new_child = Node(self._val * other._val)
            # For each input variable on which self is defined
            for input_var in self._roots.keys():
                try:
                    new_child._roots[input_var] += [(self, other._val)]
                except KeyError:
                    new_child._roots[input_var] = [(self, other._val)]

            for input_var in other._roots.keys():
                try:
                    new_child._roots[input_var] += [(other, self._val)]
                except KeyError:
                    new_child._roots[input_var] = [(other, self._val)]
            return new_child
        except AttributeError:
            new_child = Node(self._val * other)
            # For each input variable on which self is defined
            for input_var in self._roots.keys():
                try:
                    new_child._roots[input_var] += [(self, other)]
                except KeyError:
                    new_child._roots[input_var] = [(self, other)]
            return new_child


    # def __rmul__(self, other):
    #     return self * other

if __name__ == "__main__":
    # Basic demo for f(x, y) = xy + sin(x)
    x, y = Node(10), Node(20)

    x.init_roots()
    y.init_roots()

    # print(x._roots)
    # print(y._roots)

    f = y * Node.sin(x)
    f._grad_val = 1
    f.gradient(x)
    print(x._grad_val)
    print(y._val * np.cos(x._val))

    # f = x.sin()

    # f._grad_val = 1
    # f.gradient()
    # print(f"df/dx is {x._grad_val}")
    # # print(np.cos(x._val))
    # assert x._grad_val == 2 * y._val + np.cos(x._val)
    # print(f"df/dy is {y._grad_val}")
    # assert y._grad_val == 2 * x._val
