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
    
    except ZeroDivisionError:
        val = np.arcsin(x.val)
        der = float('Inf')
        return Variable(val, der)

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
    
    except ZeroDivisionError:
        val = np.arccos(x.val)
        der = float('Inf')
        return Variable(val, der)

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
    
    try:
        if x.val > 0:
            return x**0.5
        
        else:
            raise ValueError("value < 0 not valid for square root")
    
    except AttributeError:
        
        if x > 0:
            return x**0.5
    
        else:
            raise ValueError("Value < 0 not valid for square root")
