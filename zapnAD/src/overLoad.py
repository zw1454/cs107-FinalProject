import numpy as np

#overload the sin() function
def sin(x):
    #all these functions try to treat x as Variable first
    try:
        val = np.sin(x.val)
        der = np.cos(x.val) * x.der
        return Variable(val, der)

    except AttributeError():
        return np.sin(x)

#overload the cos() function
def cos(x):
    try:
        val = np.cos(x.val)
        der = -1*np.sin(x.val) * x.der
        return Variable(val, der)

    except AttributeError():
        return np.cos(x)

#overload tan() function
def tan(x):
    try:
        val = np.tan(x.val)
        der = x.der / np.cos(x.val)**2
        return Variable(val, der)
        return np.tan(x)

    except AttributeError():
        return np.tan(x)

#overload arctan() function
def arcsin(x):
    try:
        val = np.arcsin(x.val)
        der = x.der / sqrt(1-x.val**2)
        return Variable(val, der)

    except AttributeError():
        return np.arcsin(x)

#overload arccos() function
def arccos(x):
    try:
        val = np.arccos(x.val)
        der = -1 * x.der / sqrt(1-x.val**2)
        return Variable(val, der)

    except AttributeError():
        return np.arccos()

#ovrload arctan() function
def arctan(x):
    try:
        val = np.arctan(x.val)
        der = 1 / (1 + x.val**2) * sqrt(1-x.val**2)
        return Variable(val, der)

    except AttributeError():
        return np.arctan(x)
 
#overload exp() function
def exp(x):
    try:
        val = np.exp(x.val)
        der = np.exp(x.val) * x.der
        return Variable(val, der)

    except AttributeError():
        return np.exp(x)

#overload log base exp(1) function
def log(x):
    try:
        val = np.log(x.val)
        der = (1/x.val) * x.der
        return Variable(val, der)

    except AttributeError():
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

    except AttributeError():
        return np.log10(x)
  
#overload the sqrt function
def sqrt(x):
    return x**(1/2)

if __name__ == '__main__':
    from dualNumbers import *
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
