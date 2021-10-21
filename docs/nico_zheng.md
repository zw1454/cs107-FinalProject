## How to Use zapnAD

### Installation:
    pip install zapnAD

### Getting Started

First, import the the package and initialize the number of variables.

    import zapnAD as ad
    variables = ad.init_variables(1)

You can access the variables directly, or choose to relabel them to use in future equations. For example:

    x = variables[0]

Where `x` is just a refrence to `variables[0]` that you can use later. Now,let's get to the fun stuff! We can define an objective function using our variables.

    obj = ad.sin(x)

Notice, we use `ad.sin` to denote the $sin$. We also have similar elementary functions like $cos$, $sqrt$, $log$... etc overloaded for the purposes of automatic differentiation. Check the [docs](google.com) for a complete list of elementary functions included in zapnAD.

Now, that we have some objective function, we can do two things. We can evaluate the function at some `value`, or we can evaluate the derivative of the function at some `value`.

    obj.eval(1)
    obj.der(2)

### Multi-Variable Derivatives

If you are interested in the evaluation of the derivative of some function with multiple variables simply define multiple variables, and specify the which variable you would like to evaluate the derivative with respect to the other. For example:

    variables = ad.init_variables(n=2, value=(1,2))
    x, y = variables[0], variables[1]
    obj = x * ad.sin(x) + ad.cos(y**2)
    obj.der()[0]

The above code snippit will return the evaluation of the derivative `x * ad.sin(x) + ad.cos(y**2)` with respect to `x` when `x` is 1 and `y` is 2. Which is the same as solving for $\frac{\partial obj}{\partial x}(1,2)$.

## Software Orginization

### Directory Structure
The directory structure for the final project is as follows:

```
cs107-FinalProject
│   README.md
│   LICENSE  
|   requirements.txt
│
└───docs
│   │   documentation.md
|
└───zapnAD
    │   __init__.py
    │   AD.py
    |   overLoad.py
    |   dualNumbers.py
    |   variable.py
```

### Modules

Each module will server the following purpose:
 - AD.py - implement automatic differentiation foward and backward.
 - overLoad.py - overloads all elementary functions.
 - variable.py - contains abstract class for variables

### Test Suite

We plan to use TravisCI and CodeCov. CodeCov will help us track which lines of code are executed by the test suite. With travis CI we can automatically build and test code changes with each change to the package.

### Distribution and Packaging 

We will distribute our package via PyPi. In the distribution phase, we will package, build and distribute  our software using `setuptools.` To do so we will create pyproject.toml file that specifies the minimum build requirements in accordance with PEP518, and a setup.cfg file to configure static metadata.

We do not plan on using a framework for this package because frameworks are often used for more complex Python applications like website backends, dynamic web front ends and mobile clients.

## Implementation 

### Data Structures

In case that we will have to explicitly show a computation graph, we will use a dictionary to store graph structure. To implicitly define this computation graph, we will use the dual number data structure to represent different nodes in the graph. The real part of the dual number will be represent the current evaluation value, and each dual part of the dual number will represent the current derivative. The transition from one node to the next node is achieved by overloading the dual number class via one of the elementary operations.

### Classes

 - Variables - A class that contains several dual number classes based on user's specification. The number of dual numbers contained is the number of variables used to form the objective function. 
 - DualNumbers - A class that mimics the behavior of a node in the computational graph. When initialized, a dual number class is a single variable with user specified value.
 - AD - Abstract class for user to implement AD (either forward mode or backward mode).
 - ForwardMode - Abstract class to implement forward mode.
 - BackwardMode - Abstract class to implement backward mode.


### Methods and Name Attributes

The DualNumbers class will contain the following methods and attributes. We plan to overload all elementary operations to handle dual number computation under DualNumbers.
 - `__init__` will initialize the current value to be the user specified initial value via `self.value`. It will also set the initial derivative via `self.der = 1`.
 - `__add__` will add the values and derivatives by creating a new dual number class with updated attributes.
 - `__radd__` will handle the case of constant addition with a dual number.
 - `__mul__` will multiply the values and mimic the product derivative rule for derivatives by creating a new dual number class with updated attributes.
 - `__rmul__` will handle the case of constant multiplication with a dual number.
 - `__truediv__` will divide the values and mimic the division derivative rule for derivatives by creating a new dual number class with updated attributes.
 - `__pow__` will give the power of the values and mimic the power derivative rule for derivatives by creating a new dual number class with updated attributes.

For AD it would be a short method to figure out if it is using forward or backward mode and pass it to the forward or backward mode class implementations.

(Tentative) Within the foward and backward node classes:
 - Method to make the computational graph
 - method to iterate over the compuational graph
 - method to evaluate that nodes value (str8 up or derivative version)


 ### Dependencies

 We will rely on numpy to handle vector computations associated with multivariable AD. The user can easily install and check dependencies using requirements.txt in the project main directory. 

 ### Elementary Functions

To have elementary functions work on our dual number objects, we will implement them under overLoad.py so that they are now callable in the form of ad.function_name. The implementation of elementary functions will have dependency on numpy and return a new dual number object according to the rule we saw in class that evaluates a dual number. The following elementary functions will be included:
 - `ad.sin(x)`, `ad.cos(x)`, `ad.tan(x)`
 - `ad.arcsin(x)`, `ad.arccos(x)`, `ad.arctan(x)`
 - `ad.exp(x)`, `ad.log(x)`
 - `ad.pow(x, n)`