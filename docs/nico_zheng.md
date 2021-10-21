## Introduction
The main goal is to develop a software library where we can use computational methods to compute derivatives that would otherwise be costly or unstable to evaluate. Namely, we will implement automatic differentation (AD). We will implement both the forward mode and also the reverse mode. AD methods are more efficient than numerical and estimation techniques and, as was discussed in lecture, are widely applicable across a range of fields. 

## How to Use zapnAD

### Installation:
    pip install zapnAD

### Getting Started

First, import the the package and initialize the number of variables.

    import zapnAD as ad
    variables = ad.init_variables(n = 1, value = 2)

You can access the variables directly, or choose to relabel them to use in future equations. For example:

    x = variables[0]

Where `x` is just a refrence to `variables[0]` that you can use later. Now,let's get to the fun stuff! We can define an objective function using our variables.

    obj = ad.sin(x)

Notice, we use `ad.sin` to denote `sin`. We also have similar elementary functions like `cos`, `sqrt`, `log`... etc overloaded for the purposes of automatic differentiation. Check the [docs](google.com) for a complete list of elementary functions included in zapnAD.

Now, that we have some objective function, we can do two things. We can evaluate the function at some `value`, or we can evaluate the derivative of the function at some `value`.

    obj.eval()
    obj.der()

### Multi-Variable Derivatives

If you are interested in the evaluation of the derivative of some function with multiple variables simply define multiple variables, and specify the which variable you would like to evaluate the derivative with respect to the other. For example:

    variables = ad.init_variables(n=2, value=(1,2))
    x, y = variables[0], variables[1]
    obj = x * ad.sin(x) + ad.cos(y**2)
    obj.der()[0]

The above code snippit will return the evaluation of the derivative `x * ad.sin(x) + ad.cos(y**2)` with respect to `x` when `x` is 1 and `y` is 2. Which is the same as solving for <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;obj}{\partial&space;x}(1,2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;obj}{\partial&space;x}(1,2)" title="\frac{\partial obj}{\partial x}(1,2)" /></a>.

## Software Organization

### Directory Structure
The directory structure for the final project is as follows:

```
cs107-FinalProject/
│   README.md
│   LICENSE
|   setup.cfg
|   pyproject.toml
│
└───docs/
│   │   documentation.md
|
└───src/
|   |
|   └───zapnAD/
|       |   __init__.py
|       │   AD.py
|       |   overLoad.py
|       |   dualNumbers.py
|       |   variable.py
|
└───tests/

```

### Modules

Each module will server the following purpose:
 - AD.py - This module implements automatic differentiation forward and reverse mode.
 - overLoad.py - This module will overload all elementary functions.
 - variable.py - This module will contain the abstract class for handling variables in different equations.
 - dualNumbers.py - This module will contain the abstract class for handling dualNumbers.

### Test Suite

We will use pytest and pytest-cov for testing and coverage respectively. We plan to use TravisCI and CodeCov for continuous integration. 

### Distribution and Packaging 

We will distribute our package via PyPi. In the distribution phase, we will package, build and distribute  our software using `setuptools.` To do so we will create pyproject.toml file that specifies the minimum build requirements in accordance with PEP518, and a setup.cfg file to configure static metadata.

We do not plan on using a framework for this package because frameworks are often used for more complex Python applications like website backends, dynamic web front ends and mobile clients.

## Implementation 

### Data Structures

In case that we will have to explicitly show a computation graph, we will use a dictionary to store graph structure. To implicitly define this computation graph, we will use the dual number data structure to represent different nodes in the graph. The real part of the dual number will be represent the current evaluation value, and each dual part of the dual number will represent the current derivative. The transition from one node to the next node is achieved by overloading the dual number class via one of the elementary operations.

### Classes

 - Variables - A class that contains several dual number classes based on user's specification. This is to handle the case when the objective function is multivariate. The number of dual numbers contained is the number of variables used to form the objective function. 
 - DualNumbers - A class that mimics the behavior of a node in the computational graph. When initialized, a dual number class is a single variable with user specified value.
 - AD - Abstract class for user to implement AD (either forward mode or backward mode).
 - ForwardMode - Abstract class to implement forward mode.
 - ReverseMode - Abstract class to implement reverse mode.


### Methods and Name Attributes

The DualNumbers class will contain the following methods and attributes. We plan to overload all elementary operations to handle dual number computation under DualNumbers.
 - `__init__(self, value)` will initialize the current value to be the user specified initial value via `self.value`. It will also set the initial derivative via `self.der = 1`.
 - `__add__(self, other)` will add the values and derivatives by creating a new dual number class with updated attributes.
 - `__radd__(self, other)` will handle the case of constant addition with a dual number.
 - `__mul__(self, other)` will multiply the values and mimic the product derivative rule for derivatives by creating a new dual number class with updated attributes.
 - `__rmul__(self, other)` will handle the case of constant multiplication with a dual number.
 - `__truediv__(self, other)` will divide the values and mimic the division derivative rule for derivatives by creating a new dual number class with updated attributes.
 - `__pow__(self, other)` will give the power of the values and mimic the power derivative rule for derivatives by creating a new dual number class with updated attributes.

For AD it would be a short method to figure out if it is using forward or reverse mode and pass it to the forward or backward mode class implementations.
 - `forward()`: Use forward mode to evaluete the derivative of the objective function.
 - `reverse()`: Use reverse mode to evaluete the derivative of the objective function.

(Tentative) Within the foward and backward node classes:
 - Method to make the computational graph
 - method to iterate over the compuational graph
 - method to evaluate that nodes value (str8 up or derivative version)


 ### Dependencies

 We will rely on numpy to handle vector computations associated with multivariable AD as well as overloading the elementary functions outlined below. We will specify the dependencies for the package in setup.cfg. 

 ### Elementary Functions

To have elementary functions work on our dual number objects, we will implement them under overLoad.py so that they are now callable in the form of ad.function_name. The implementation of elementary functions will have dependency on numpy and return a new dual number object according to the rule we saw in class that evaluates a dual number. The following elementary functions will be included:
 - `ad.sin(x)`
 - `ad.cos(x)`
 - `ad.tan(x)`
 - `ad.arcsin(x)`
 - `ad.arccos(x)`
 - `ad.arctan(x)`
 - `ad.exp(x)`
 - `ad.log(x)`
 - `ad.pow(x, n)`

## Licensing
We have decided to choose the GNU General Public License. We chose this license because it is a copyleft license.  Copyleft allows users to use and modify our software and, as stated on the GNU GPL website, says that "anyone who redistributes the software, with or without changes, must pass along the freedom to further copy and change it." As beneficiaries of free software, we would like to makes ours free as well. 

More on this particular license can be found here: https://www.gnu.org/licenses/gpl-3.0.html 
