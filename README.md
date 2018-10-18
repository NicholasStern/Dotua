# AutoDiff Documentation (Milestone 1)

### Nick Stern, Vincent Viego, Summer Yuan, Zach Wehrwein

## Introduction
Calculus, according to the American mathematician Michael Spivak in his noted textbook, is fundamentally the study of "infinitesimal change." An infinitesimal change, according to Johann Bernoulli as Spivak quotes, is so tiny that "if a quantity is increased or decreased by an infinitesimal, then that quantity is neither increased nor decreased." The study of these infinitesimal changes is the study of relationships of change, not the computation of change itself. The derivative is canonically found as function of a limit of a point as it approaches 0 -- we care about knowing the relationship of change, not the computation of change itself.

One incredibly important application of the derivative is varieties of optimization problems. Machines are able to traverse gradients iteratively through calculations of derivatives. However, in machine learning applications, it is possible to have millions of parameters for a given neural net and this would imply a combinatorially onerous number of derivatives to analytically calculate. A numerical Newton's method approach (iteratively calculating through guesses of a limit) is likewise not a wise alternative because even "very small" $h$, the end result can be orders of magnitude off in error relative to machine precision.

So might think that a career in ML thus requires an extensive calculus background, but, Ryan P Adams, formerly of Twitter (and Harvard IACS), now of Princeton CS, describes automatic differentiation as ["getting rid of the math that gets in the way of solving a [ML] problem."](https://www.youtube.com/watch?v=sq2gPzlrM0g) What we ultimately care about is tuning the hyperparameters of a machine learning algorithm, so if we can get a machine to do this for us, that is ultimately what we care about. What is implemented in this package is automatic differentiation which allows us to calculate derivatives of complex functions to machine precision 'without the math getting in the way.'

## Background
The most important calculus derivative rules for automatic differentiation is the multivariate chain rule.

The basic chain rule states that the derivative of a composition of functions is:

<img src="https://latex.codecogs.com/png.latex?(f&space;\circ&space;g)^{'}&space;=&space;(f^{'}&space;\circ&space;g)&space;\cdot&space;g^'" title="(f \circ g)^{'} = (f^{'} \circ g) \cdot g^'" />

That is, the derivative is a function of the incremental change in the other multiplied by the inner function, multiplied by the change in the outer.

In the multivariate case, we can apply the chain rule as well as the rule of total differentiation. For instance, if we have a simple equation:

<img src="https://latex.codecogs.com/png.latex?y&space;=&space;u&space;\cdot&space;v" title="y = u \cdot v" />

Then,

<img src="https://latex.codecogs.com/png.latex?y&space;=&space;f(u,v)" title="y = f(u,v)" />

The partial derivatives:

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;y}{\partial&space;u}&space;=&space;v" title="\frac{\partial y}{\partial u} = v" />

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;y}{\partial&space;v}&space;=&space;u" title="\frac{\partial y}{\partial v} = u" />

The total variation of y depends on both the variations in u, v and thus,

<img src="https://latex.codecogs.com/png.latex?dy&space;=&space;\frac{\partial&space;y}{\partial&space;u}&space;du&space;&plus;&space;\frac{\partial&space;y}{\partial&space;v}&space;dv" title="dy = \frac{\partial y}{\partial u} du + \frac{\partial y}{\partial v} dv" />

What this trivial example illustrates is that the derivative of a multivariate function is ultimately the addition of the partial derivatives and computations of its component variables. If a machine can compute any given sub-function as well as the partial derivative between any sub-functions, then the machine need only add-up the production of a function and its derivatives to calculate the total derivative.

[An intuitive way of understanding automatic differentiation is to think of any complicated function as ultimately a a graph of composite functions.](http://colah.github.io/posts/2015-08-Backprop/) Each node is a primitive operation -- one in which the derivative is readily known -- and the edges on this graph -- the relationship of change between any two variables -- are partial derivatives. The sum of the paths between any two nodes is thus the partial derivative between those two functions (this a graph restatement of the total derivative via the chain rule).

Forward mode automatic differentiation thus begins at an input to a graph and sums the paths feeding. In the below diagrams (from Christopher Olah's blog) provide an intuition for this process. The relationship between three variables (X, Y, Z) is defined by a number of paths ($\alpha, \beta, \gamma, \delta, \epsilon ,\zeta $). Forward mode begins with a seed of 1, and then in each node derivative is the product of the sum of the previous steps.

![](images/chain-def-greek.png)

![](images/chain-forward-greek.png)

Consequently, provided that within each node there is an elementary function, the machine can track the derivative through the computational graph.

There is one last piece of the puzzle: dual numbers which extend the reals by restating each real as $x + epsilon$, where $epsilon^2 = 0$. Symbolic evaluation, within a machine, can quickly become computational untenable because the machine must hold in memory variables and their derivatives in the successive expansions of the rules of calculus. Dual numbers allow us to track the derivative of an even a complicated function, as a kind of data structure that caries both the derivative and the primal of a number.

In our chain rule equation, there are two pieces to the computation: the derivative of the outer function multiplied by the inner function and that product multiplied by the derivative of the inner. This means that the full symbolic representation of an incredibly complicated function can stretch to exponentially many terms. However, dual numbers allow us to parse that symbolic representation in bitesized pieces that can be analytically computed. [The reason for this is the Taylor series expansion of a function](http://jliszka.github.io/2013/10/24/exact-numeric-nth-derivatives.html):

<img src="https://latex.codecogs.com/png.latex?f(x&plus;p)&space;=&space;f(x)&space;&plus;f'(x)p&space;&plus;&space;\frac{f''(x)p^2}{2!}&space;&plus;&space;\frac{f^{(3)}(x)p^3}{3!}" title="f(x+p) = f(x) +f'(x)p + \frac{f''(x)p^2}{2!} + \frac{f^{(3)}(x)p^3}{3!}" />

When one evaluates <img src="https://latex.codecogs.com/png.latex?f(x&space;&plus;&space;\epsilon)" title="f(x + \epsilon)" />, given that <img src="https://latex.codecogs.com/png.latex?\epsilon^2&space;=&space;0" title="\epsilon^2 = 0" />, then all the higher order terms drop and one is left with <img src="https://latex.codecogs.com/png.latex?f(x)&space;&plus;f'(x)\epsilon" title="f(x) +f'(x)\epsilon" />


To recap: automatic differentiation is an algorithmic means of computing complex derivatives by parsing those functions as a graph structures to be traversed. Dual numbers are used as a sort of mathematical data structure which allows the machine to analytically compute the derivative at any given node. It is superior to analytic or symbolic differentiation because it actually is possible to do! And it is superior to numerical methods because automatic differentiation is far more accurate. 

## How to Use AutoDiff
In order to instantiate an auto-differentiation object from our package, the user shall first import the AutoDiff library:

```py
Import AutoDiff as ad
```

The general workflow for the user is as follows:
- Instantiate all variables as AutoDiff “Variable” objects.
- Input these variables into elementary functions from the AutoDiff library to create more complex expressions that propagate the derivative. 

The “Variable” class is the core constructor for all variables in the function that are to be differentiated.  The general schematic for how the user shall instantiate and interact with this class is outlined below:

1. Create an “Variable” object out of each variable in the function to be differentiated,  passing in the value of the variable as follows:

```python
x = ad.Variable(value = 1)
```

- To instantiate multiple “Variable” objects at once, the user may pass in a list of values:

```python
var_list = ad.Variable(value = [1, 2, 3])
```

2. Next, the user shall pass in these variables into elementary functions as follows:
```python
result = ad.sin(x)
results = ad.sin(var_list)
```

Simple operators, such as sums and products, can be used normally:
```python
result = 6*x
results = var_list + 4
```

```python
x, y, z = ad.Variable(value = [2, 3, 4])
result = ad.sin(x) + 6*y + x*y*z
```
	
3. Finally, (as a continuation of the previous example), the user may access the value and derivative of a function using the following attributes:

```python
result.val # The value of the function
result.der # The derivative of the function
```

## Software Organization
To be filled in by Vincent

## Implementation
As for implementation of the forward mode of automatic differentiation, there are 5 steps to consider:

	1. The user generates the initial variables.
	2. The user generates the goal function.
	3.  The program constructs the computational graph according to the goal function.
	4. The program completes the computations in the computational graph.
	5. The program returns the final result to the user's goal function.
	
To construct the computational graph, the core data structures should clearly represent all the nodes and edges included in the graph.
So we have a superclass Node, which can describe all kinds of nodes included in the graph. For the superclass Node, there are 3 subclasses: variable, constant, complex (not basic variables or constants) Node. 
For each Node, we have previous Nodes (inputs), operator, constant and expression as its attributes. Previous Nodes are the Nodes who have edges pointing to the Node. Operator is the operation to get the Node if the Node is not the initial variable itself. The attribute 'constant' will denote whether this node is a constant. 

```python
class Node():
	def __init__(self):
		self.inputs = []
		self.operator = None
		self.constant = False
		self.expression = ''

class Variable(Node):
	def __init__(self, name = 'x', val):
		self.inputs = []
		self.operator = PlaceHolder
		self.constant = False
		self.expression = name
		self.value = val

class Constant(Node):
	def __init__(self, number):
		self.inputs = []
		self.operator = PlaceHolder
		self.constant = True
		self.expression = str(value)
		self.value = number

class Complex(Node):
	def __init__(self, op, inputs, func):
		self.inputs = inputs
		self.operator = op
		self.constant = False
		self.expression = func
```

After the class node is created, the user can generate initial variables. They may do this by:

```python
x1 = ad.Variable('x1',2)
x2 = ad.Variable('x2',3)
```

Then the user wants to generate the goal function, which needs to load elementary functions such as $sin, exp, add$. We will create another class Operator to help the user with loading the operators/elementary functions. 
Operator is a superclass, and we will create subclasses which include all categories of operators/elementary functions which are important. (We may overload some basic operations.) We will also consider the problems of vector inputs when implementing the subclasses. Also there is a subclass with no computational functions, which can be given to the initial variables since there are no edges pointing to them.
For each Operator, it will return a new Node class which includes the previous nodes and itself as the operator attribute. We have methods of getting values and getting derivatives for Operator classes, which can be called by giving previous nodes as inputs. For different operators/elementary functions subclasses, there will be different restrictions. Also, we will rely on numpy to get the value of the calculation results of these operations.

```python
class Operator():
	def __call__(self, inputs = []):
		next_node = Node()
		next_node.inputs = inputs
		next_node.operator = self
		return next_node

	def get_value(self, node, val):
		raise NotImplementedError

	def get_gradient(self, node, val):
		raise NotImplementedError
```

Then the user can construct the goal function by calling the Node classes and the Operator classes. For example:

```python
y = ad.sin(x1+x2)
```


![](images/classes.png)

Now that we have the goal function, we need to construct the whole computational graph and get the right order to do the calculations. To do this, we use a binary tree data structure to store the goal function. We just read the user's inputs from right to left, but every time when we encouter a new object (Node or Operator), we will put them in a binary tree. For every operator, it will be compared to the root node of the current binary tree. If its rank is higher than that of the root node, then it becomes the right child of the root node, and the original right tree of the root node will be the operator's right tree. If its rank is not higher that of the root node, then it becomes the root node, the original tree will be the operator's left tree. For every Node object, we just put it into the right child of the right tree.


![](images/binary_tree.png)

To construct the computational graph, we create a new class Graph, and it should be called when users want to get values or derivatives of a function (it may be regarded as a Complex object in our library).

```python
class Graph():
	def __init__(self, func):
		self.func = func
		[The process of building up the binary tree, the first node of the tree will be an attribute of the Graph, it will also keep a list of Variables, Complexes in the func as an attribute of the Graph]
	def get_val_gradient(self):
		[It will Preorder Trasversal the bianry tree and set up a stack to store the visited but not yet calculated Nodes and Operators; For every Operator, it will call get_value to get the value of the new Node, and it will call get_gradient to get partial differentials of every Variable or Complex in the list of the Graph; It will return a dictionary of value and the partial differentials of every Variable or Complex included in the function]
```

When Graph.get\_val\_gradient() is called, we use PreorderTraversal to visit the whole binary tree to complete the calculation. We will use a stack to store the nodes. Every time when we press a new node into the stack, we will try to see if there are enough inputs to feed the first Operator in the stack to return a new Node. Every time when a new Node is returned by the operator, we will put the new Node into the stack. When all the nodes in the binary tree are visited and there is only one Node (the Complex Node that we want to evaluate) in the stack, we can get the values/derivatives of the goal function.
![](images/calculating_process.png)

And the user will just do the following things to get value and gradients of the goal function:

```python
y_graph = Graph(y)
y_dict = y_graph.get_val_gradient()
y_val = y_dict['value']
dy/dx1 = y_dict['x1']
```
