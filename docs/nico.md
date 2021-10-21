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

Because we know that we will have to implicietly define a computational graph,we will use a dictionaireis to store graph structure. Each key of the node would be a trace, and each value would have a list of triples that consists of the evaluation value, the elementary function, and the next node.

### Classes
 - Dual Numbers - A class that handles operations concerning dual numbers.
 - AD - Abstract class for user to implement AD (either fowardmode or backward mode).
 - FowardMode - Abstract class to implement foward mode.
 - BackwardMode - Abstract calss toimplement backaward mode.


 ### Methods and Name Attributes

We plan to overload all elementary functions to handle dual number computation and work with functions like $cos$, $sin$, $tan$ $sqrt$, $power$ $log$, $exp$, and probably more.

For AD it would be a short method to figure out if its foward or backward node and pass it to the foward or backward mode class implementations.

Within the foward and backward node classes:
 - Method to make the compuational graph
 - method to iterate over the compuational graph
 - method to eval that nodes value (str8 up or derivative version)


 ### Dependencies

 We will rely on numpy to handle vector computations associated with multivariable AD. The user can easily install and check dependencies using requirements.txt in the project main directory. 

