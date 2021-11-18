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

#overload the sin() function
def sin(x):
    #all these functions try to treat x as Variable first
    try:
        val = np.sin(x.val)
        der = np.cos(x.val) * x.der
        return Variable(val, der)

    except:
        return np.sin(x)

#overload the cos() function
def cos(x):
    try:
        val = np.cos(x.val)
        der = -1*np.sin(x.val) * x.der
        return Variable(val, der)

    except:
        return np.cos(x)

#overload tan() function
def tan(x):
    try:
        val = np.tan(x.val)
        der = x.der / np.cos(x.val)**2
        return Variable(val, der)
        return np.tan(x)

    except:
        return np.tan(x)

#overload arctan() function
def arcsin(x):
    try:
        val = np.arcsin(x.val)
        der = x.der / sqrt(1-x.val**2)
        return Variable(val, der)

    except:
        return np.arcsin(x)

#overload arccos() function
def arccos(x):
    try:
        val = np.arccos(x.val)
        der = -1 * x.der / sqrt(1-x.val**2)
        return Variable(val, der)

    except:
        return np.arccos()

#ovrload arctan() function
def arctan(x):
    try:
        val = np.arctan(x.val)
        der = 1 / (1 + x.val**2) * sqrt(1-x.val**2)
        return Variable(val, der)

    except:
        return np.arctan(x)
 
#overload exp() function
def exp(x):
    try:
        val = np.exp(x.val)
        der = np.exp(x.val) * x.der
        return Variable(val, der)

    except:
        return np.exp(x)

#overload log base exp(1) function
def log(x):
    try:
        val = np.log(x.val)
        der = (1/x.val) * x.der
        return Variable(val, der)

    except:
        return np.log(x)

#overload log base 2 function
def log2(x):
    try:
        val = np.log2(x.val)
        der = (1/(x.val *np.log(2))) * x.der
        return Variable(val, der)
    #for some reason it doesnt like except AttributeErrors here    
    except:
        return np.log2(x)

#overload log base 10 function
def log10(x):
    try:
        val = np.log10(x.val)
        der = (1/(x.val *np.log(10))) * x.der
        return Variable(val, der)

    except:
        return np.log10(x)
  
#overload the sqrt function
def sqrt(x):
    return x**(1/2)

if __name__ == '__main__':
    variables = Variables(n=1).set_values(values=[3]) # List of object Variable
    x = variables[0]
    y = (2 * x + x ** 2)
    print(y.der)
    print(y.val)
    print((y ** 2).der)
