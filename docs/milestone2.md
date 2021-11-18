## Introduction
The main goal is to develop a software library where we can use computational methods to compute derivatives that would otherwise be costly or unstable to evaluate. Namely, we will implement automatic differentation (AD). We will implement both the forward mode and also the reverse mode. AD methods are more efficient than numerical and estimation techniques and, as was discussed in lecture, are widely applicable across a range of fields.

Our work is significant mainly in two aspects. Firstly, the client's need of computing derivative for real-world applications is high, which has been a long tradtion in the applied sciences domains like mechanical engineering and mathematical physics. Moreover, the recent advances in data science and machine learning enables more sophisticated models like deep neural networks, whose large number of parameters require more efficient algorithm like back-propagation to compute and update the gradients. With our AD library, researchers in these fields will be able to efficiently compute the gradient through a simple and interactive user interface.

Secondly, although many software packages support numerical approaches like Newton's method and finite-element method, they often lack numerical accuracy due to approximation and are computationally expensive. Our AD approach overcomes these issues by efficiently computing the exact, symbolic form of the derivative, which is crucial for real-world engineering problems.

## Background

Automatic differentiation enables us to take the derivative of arbitrarily complex functions *f* of a given independent variable *x* in a way that is computationally cost-effective and numerically accurate. 

### The Chain Rule
At the heart of automatic differentiation is the chain rule for derivatives, in which the derivatives of compositions of functions can be written as the product of the derivatives of nested functions: 

<img src="https://latex.codecogs.com/gif.latex?\fn_phv&space;f(g(x))&space;:&space;\frac{\mathrm{d}f}{\mathrm{d}x}=\frac{\mathrm{d}&space;f}{\mathrm{d}&space;g}\frac{\mathrm{d}&space;g}{\mathrm{d}&space;x}" title="f(g(x)) : \frac{\mathrm{d}f}{\mathrm{d}x}=\frac{\mathrm{d} f}{\mathrm{d} g}\frac{\mathrm{d} g}{\mathrm{d} x}" />

When the function *f* has multiple inputs, then the derivative of *f* is the sum of the derivatives of *f* with respect to each of the inputs. In the case where the independent variable *x* has only one dimension, this is:

<img src="https://latex.codecogs.com/gif.latex?\fn_phv&space;\begin{align*}&space;f(g_{1}(x),&space;g_{2}(x),&space;...,&space;g_{n}(x)):&space;\frac{\mathrm{d}f}{\mathrm{d}x}&space;&=&space;\sum_{i=1}^{n}\frac{\mathrm{d}f}{\mathrm{d}g_{i}}\frac{\mathrm{d}g_{i}}{\mathrm{d}x}&space;\end{align*}" title="\begin{align*} f(g_{1}(x), g_{2}(x), ..., g_{n}(x)): \frac{\mathrm{d}f}{\mathrm{d}x} &= \sum_{i=1}^{n}\frac{\mathrm{d}f}{\mathrm{d}g_{i}}\frac{\mathrm{d}g_{i}}{\mathrm{d}x} \end{align*}" />

### Forward Mode AD
We can write a function *f* as a partial ordering of elementary operations starting with the independent variable *x*. For example:

<img src="https://latex.codecogs.com/gif.latex?\fn_phv&space;\begin{align*}&space;f(x)&space;&=&space;\log(\sin(x)&space;&plus;&space;4x)&space;\\&space;&=g_{4}(g_{3}(g_{2}(x),&space;g_{1}(x)))\\&space;\text{With&space;the&space;following&space;intermediate&space;elementary&space;functions}&space;\\&space;g_{1}(u)&space;&=&space;\sin(u)&space;\\&space;g_{2}(u)&space;&=&space;4u&space;\\&space;g_{3}(u,v)&space;&=&space;u&space;&plus;&space;v&space;\\&space;g_{4}(u)&space;&=&space;\log(u)&space;\\&space;\end{align*}" title="\begin{align*} f(x) &= \log(\sin(x) + 4x) \\ &=g_{4}(g_{3}(g_{2}(x), g_{1}(x)))\\ \text{With the following intermediate elementary functions} \\ g_{1}(u) &= \sin(u) \\ g_{2}(u) &= 4u \\ g_{3}(u,v) &= u + v \\ g_{4}(u) &= \log(u) \\ \end{align*}" />


These elementary functions are combined in a single direction, meaning that once an intermediate value (represented as variables g<sub>0</sub>, g<sub>1</sub>, g<sub>2</sub>...) is calculated, the previous values do not need to be saved. The function *f* can be evaluated at a particular *x* by stepping through the elementary functions in the proper order. This is called the primal trace. For this example:

#### Primal Trace
<img src="https://latex.codecogs.com/gif.latex?\fn_phv&space;\begin{align*}&space;g_{0}&space;&=&space;x_{1}&space;\\&space;g_{1}&space;&=&space;\sin(g_{0})&space;\\&space;g_{2}&space;&=&space;4g_{0}&space;\\&space;g_{3}&space;&=&space;g_{1}&space;&plus;&space;g_{2}&space;\\&space;g_{4}&space;&=&space;\log(g_{3})&space;=&space;f(x)\\&space;\end{align*}" title="\begin{align*} g_{0} &= x_{1} \\ g_{1} &= \sin(g_{0}) \\ g_{2} &= 4g_{0} \\ g_{3} &= g_{1} + g_{2} \\ g_{4} &= \log(g_{3}) = f(x)\\ \end{align*}" />

The derivative with respect to the independent variable *x* can also be computed by stepping through the computational graph. This works "inside out" from the independent variable *x* to the final arbitrarily complex function *y*.  Each of these elementary operations has a simple, known derivative that can be quickly accessed or calculated. Taken together, the computational graph's unidirectionality and insight from the chain rule shows that the derivative at each intermediate step only requires (a) knowledge of the value of the function (the primal trace) and of the derivative (called the tangent trace) from the step immediately prior (the 'parent'; could be multiple if the current function takes multiple inputs, like addition) and (b) the elementary function (and its derivative) at the current step. This is the "Forward Mode" for AD, which will produce both the intermediate values and directional derivatives of the function *f* with respect to *x*. For the above example, the traces calculated are as follows. The D<sub>p</sub> represent directional derivatives, that is, derivatives with respect to a particular independent variable:

#### Tangent Trace
<img src="https://latex.codecogs.com/gif.latex?\fn_phv&space;\begin{align*}&space;D_{p}g_{0}&space;&=&space;1&space;\\&space;D_{p}g_{1}&space;&=&space;\frac{\mathrm{d}g_{1}}{\mathrm{d}g_{0}}&space;=&space;cos(g_{0})D_{p}g_{0}\\&space;D_{p}g_{2}&space;&=\frac{\mathrm{d}g_{2}}{\mathrm{d}g_{0}}&space;=&space;4D_{p}g_{0}&space;\\&space;D_{p}g_{3}&space;&=&space;\frac{\mathrm{d}g_{3}}{\mathrm{d}g_{1}}D_{p}g_{1}&space;&plus;&space;\frac{\mathrm{d}g_{3}}{\mathrm{d}g_{2}}D_{p}g_{2}&space;=&space;D_{p}g_{1}&space;&plus;&space;D_{p}g_{2}&space;\\&space;D_{p}g_{4}&space;&=\frac{\mathrm{d}g_{4}}{\mathrm{d}g_{3}}&space;\log(g_{3})&space;=&space;log(g_{3})D_{p}g_{3}&space;\\&space;\end{align*}" title="\begin{align*} D_{p}g_{0} &= 1 \\ D_{p}g_{1} &= \frac{\mathrm{d}g_{1}}{\mathrm{d}g_{0}} = cos(g_{0})D_{p}g_{0}\\ D_{p}g_{2} &=\frac{\mathrm{d}g_{2}}{\mathrm{d}g_{0}} = 4D_{p}g_{0} \\ D_{p}g_{3} &= \frac{\mathrm{d}g_{3}}{\mathrm{d}g_{1}}D_{p}g_{1} + \frac{\mathrm{d}g_{3}}{\mathrm{d}g_{2}}D_{p}g_{2} = D_{p}g_{1} + D_{p}g_{2} \\ D_{p}g_{4} &=\frac{\mathrm{d}g_{4}}{\mathrm{d}g_{3}} \log(g_{3}) = log(g_{3})D_{p}g_{3} \\ \end{align*}" />

This can be extended in two ways:
1. The independent variable *x* can have multiple dimensions *m*. 
2. The function *f* can have multiple dimensions.

In a multidimensional setting, g<sub>-m</sub>...<sub>0</sub> represent the independent variables and D<sub>p</sub>g<sub>-j</sub> is a vector, which specifies the independent variable of interest. In forward mode, one must traverse (implicitly or explicitly) the computational graph for each independent variable to compute the full gradient of *f*. This becomes computationally infeasible in settings with very large *m*, motivating reverse mode (below).

### Reverse Mode AD
In order to instead calculate the partial derivatives of *f* with respect to the independent variable *x* and the intermediate dependent variables *g<sub>i</sub>* (for example, to determine the sensitivity of *f* to that particular intermdiate), one can traverse backwards through the graph. This derivative of *f* with respect to a particular *g<sub>i</sub>* is called the adjoint of *g<sub>i</sub>*. The reverse mode requires two passes:

1. Forward pass: compute the primal trace (as above) and compute the partial derivatives of each child node with respect to its parent node. These (numeric) values have to be stored, which makes reverse mode more space intensive. 
2. Reverse pass: the graph is traversed from outputs (*f*) towards inputs and each adjoint is calculated in succession using the stored values of the intermediate nodes and their partial derivatives.  

Importantly, the gradient of *f* computed by forward mode (the derivatives of *f* with respect to each of the independent variables) is the same as the first *m* adjoints computed by the reverse mode.

## How to Use zapnAD

**THIS NEEDS TO BE REWRITTEN**

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

**WE NEED TO DESCRIBE THE TEST SUITE**

### Distribution and Packaging 

**THIS WILL CHANGE NOW BECAUSE IT IS NOT ON PYPI**

## Implementation 

### Data Structures

To implicitly define a computational graph, we used the dual number data structure to represent different nodes in the graph. The real part of the dual number represents the current evaluation value, and each dual part of the dual number will represent the current derivative. The transition from one node to the next node is achieved by overloading the dual number class via one of the elementary operations.

### Classes

 - Variables - A class that contains several dual number classes based on user's specification. This is to handle the case when the objective function is multivariate. The number of dual numbers contained is the number of variables used to form the objective function. 
 - Variable - A class that mimics the behavior of a node in the computational graph. When initialized, a dual number class is a single variable with user specified value.

### Methods and Name Attributes

The Variable class will contain the following methods and attributes. We plan to overload all elementary operations to handle dual number computation under DualNumbers.
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
**HOW DO WE ACTUALLY USE THE REQUIREMENTS.txt**

 ### Elementary Functions

To have elementary functions work on our dual number objects, we included elementary functions in overLoad.py so that they are now callable in the form of ad.function_name.  

## Licensing
This software is licensed under the GNU General Public License. This Copyleft license allows users to use and modify our software and, as stated on the GNU GPL website, says that "anyone who redistributes the software, with or without changes, must pass along the freedom to further copy and change it." As beneficiaries of free software, we would like to makes ours free as well. 

More on this particular license can be found here: https://www.gnu.org/licenses/gpl-3.0.html 

## Future Features
