import numpy as np
from .dualNumbers import Variable, Variables

__all__ = ['sin', 'cos','tan','arcsin', 'arccos', 'arctan', 'exp', 'log', 'log2',
        'log10', 'sqrt', 'sinh', 'cosh', 'tanh']

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
        der = -1 * x.der / sqrt(1.0 - x.val**2)
        return Variable(val, der)

    except AttributeError:
        return np.arccos(x)


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
        der = x.der / (1.0 + x.val**2)
        return Variable(val, der)

    except AttributeError:
        return np.arctan(x)
    
 
def exp(x, base=None):
    '''
    Overloads the exp() functions. If base not specified, handles
    the natural base e as the special case. If base is specified, 
    then treat the base as such and calculate the exponential 
    accordingly. 
    
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    if base is None: 
        try:
            val = np.exp(x.val)
            der = np.exp(x.val) * x.der
            return Variable(val, der)

        except AttributeError:
            return np.exp(x)
    else: 
        try: 
            val = base**x.val 
            der = (x.val* base**(x.val - 1)) * x.der 
            return Variable(val, der)

        except AttributeError:
            return base**x


def log(x, base=None):
    '''
    If base is None, overloads the natural logarithm ln() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number

    If base is not None, and is some other base, returns the log of 
    that base function for the Variable object, otherwise defaults 
    on it being a real number and apply the operation similarly. 
    '''        
    if base is None: # natural number case:
        try:
            val = np.log(x.val)
            der = (1/x.val) * x.der
            return Variable(val, der)

        except AttributeError:
            return np.log(x)
    
    else: 
        if base < 1:
            raise ValueError("Log base must be greater than or equal to 1")
        try: 
            val = np.log(x.val) / np.log(base)
            der = (1/(x.val * np.log(base))) * x.der
            return Variable(val, der)

        except AttributeError:
            return np.log(x) / np.log(base)

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
    
    try:
        val = x.val
    
    except AttributeError:
        val = x
        
    if val >= 0:
        return x**0.5

    else:
        raise ValueError("Value < 0 not valid for square root")

def sinh(x): 
    '''
    Creates the sinh() function. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.sinh(x.val)
        der = np.cosh(x.val) * x.der
        return Variable(val, der)

    except AttributeError:
        return np.sinh(x)
    
def cosh(x):
    '''
    Overloads the cosh() function. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.cosh(x.val)
        der = np.sinh(x.val) * x.der
        return Variable(val, der)

    except AttributeError:
        return np.cosh(x)

def tanh(x):
    '''
    Overloads the tanh() functions. 
    The function will first try to treat x as a Variable object.
    Returns:
        - A new Variable object if x is a Variable object
        - A real number if x is a real number
    '''
    try:
        val = np.tanh(x.val)
        der = (1 - (np.tanh(x.val))**2) * x.der 
        return Variable(val, der)

    except AttributeError:
        return np.tanh(x)
