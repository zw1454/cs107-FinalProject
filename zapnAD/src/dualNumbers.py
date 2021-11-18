import numpy as np 

class Variable:
    def __init__(self, value, der = 1) -> None:
        self.val = value
        self.der = der

    def __str__(self):
        return f"Dual Number: Value {self.val}, Derivative: {self.der}."
    
    def __mul__(self, other):
        # Product derivative rule for two Variable types
        try:
            new_f = Variable(self.val * other.val)
            new_f.der = self.der * other.val + self.val * other.der
        # When other is a real number
        except AttributeError:
            new_f = Variable(self.val * other)
            new_f.der = self.der * other
        return new_f
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        # Sum of derivatives for two Variable types
        try:
            new_f = Variable(self.val + other.val)
            new_f.der = self.der + other.der
        # When other is a real number (beta)
        except AttributeError:
            new_f = Variable(self.val + other)
            new_f.der = self.der
        return new_f
    
    #When we use the "-" operator dunder
    def __neg__(self):
        return Variable(-1*self.val, -1*self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        #Subtraction using the dunder methods above
        return self + (-1*other)

    def __rsub__(self, other):
        return (-1*other) + self
    
    def __pow__(self, p):
        new_f = Variable(self.val ** p)
        new_f.der = p * self.val ** (p - 1) * self.der
        return new_f

class Variables:
    def __init__(self, n):
        self.n = n
    
    def set_values(self, values):
        assert len(values) == self.n, 'Dimension Mismatch!'
        return [Variable(value) for value in values]

#overload elem functions

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

    except AttributeError():
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

    except AttributeError():
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

    except AttributeError():
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

    except AttributeError():
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

    except AttributeError():
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

    except AttributeError():
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

    except AttributeError():
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

    except AttributeError():
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
    except:
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

    except AttributeError():
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
    print(y.der)
    print(y.val)
    print((y ** 2).der)
