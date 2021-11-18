import numpy as np
from .dualNumbers import Variable, Variables

def sin(x):
    '''
    Overloads the sin() function. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.sin(x.val)
        der = np.cos(x.val) * x.der
        return Variable(val, der)

    except AttributeError:
        return np.sin(x)

def cos(x):
    '''
    Overloads the cos() function. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.cos(x.val)
        der = -1 * np.sin(x.val) * x.der
        return Variable(val, der)

    except AttributeError:
        return np.cos(x)

def tan(x):
    '''
    Overloads the tan() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.tan(x.val)
        der = x.der / np.cos(x.val)**2
        return Variable(val, der)

    except AttributeError:
        return np.tan(x)

def arcsin(x):
    '''
    Overloads the arcsin() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.arcsin(x.val)
        der = x.der / sqrt(1 - x.val**2)
        return Variable(val, der)

    except AttributeError:
        return np.arcsin(x)

def arccos(x):
    '''
    Overloads the arccos() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.arccos(x.val)
        der = -1 * x.der / sqrt(1 - x.val**2)
        return Variable(val, der)

    except AttributeError:
        return np.arccos()

def arctan(x):
    '''
    Overloads the arctan() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.arctan(x.val)
        der = x.der / (1 + x.val**2)
        return Variable(val, der)

    except AttributeError:
        return np.arctan(x)
 
def exp(x):
    '''
    Overloads the exp() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.exp(x.val)
        der = np.exp(x.val) * x.der
        return Variable(val, der)

    except AttributeError:
        return np.exp(x)

def log(x):
    '''
    Overloads the natural logarithm ln() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.log(x.val)
        der = (1/x.val) * x.der
        return Variable(val, der)

    except AttributeError:
        return np.log(x)

def log2(x):
    '''
    Overloads the log2() (base 2) functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.log2(x.val)
        der = (1/(x.val * np.log(2))) * x.der
        return Variable(val, der)
    # For some reason it doesn't like except AttributeErrors here    
    except AttributeError:
        return np.log2(x)

def log10(x):
    '''
    Overloads the log10() (base 10) functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.log10(x.val)
        der = (1/(x.val * np.log(10))) * x.der
        return Variable(val, der)

    except AttributeError:
        return np.log10(x)
  
def sqrt(x):
    '''
    Overloads the sqrt() functions. 
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    return x**(1/2)

if __name__ == '__main__':
    variables = Variables(n=1).set_values(values=[3]) # List of object Variable
    x = variables[0]
    y = (2 * x + x ** 2)
    #arcsin and arccos are returning warnings. Note their value becomes NaN
    print(y)
    print(sin(y))
    print(cos(y))
    print(log2(2))
    print(log(y))
    print(sqrt(16))
    print(sqrt(y))
    print(arccos(y))
    print(arcsin(y))
    print(arctan(y))
    print(tan(y))
