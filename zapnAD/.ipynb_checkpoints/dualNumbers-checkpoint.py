import numpy as np

class Variable:
    def __init__(self, value, derivatives=None) -> None:
        '''
        Stores the current value and derivative of this variable.

        Input:
            - self.val: int or float, current value
            - self.der: ndarray, full derivative
        '''
        self.val = value
        self.der = derivatives

    def __str__(self):
        '''
        Returns:
            - str, String representation of the variable
        '''
        return f"Value {self.val}\n" + \
            f"Full Jacobian {self.der}\n"
    
    def get_value(self):
        '''
        Returns:
            - int or float, current value of the variable
        '''
        return self.val
    
    def get_gradient(self):
        '''
        Returns:
            - ndarray of size (n, ), current full derivative of the variable (Jacobian)
        '''
        return self.der
    
    def __mul__(self, other):
        '''
        Compute the new variable with updated value and derivative after multiplication.

        Input:
            - other: int or float or Variable instance
        
        Returns:
            - Variable instance
        '''
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
        '''
        Special dunder method to handle the case of int/float * Variable instance.
        Convert to Variable * int/float and then handled by __mul__(self, other).

        Input:
            - other: int or float or Variable instance
        
        Returns:
            - Variable instance
        '''
        return self.__mul__(other)
    
    def __add__(self, other):
        '''
        Compute the new variable with updated value and derivative after addition.

        Input:
            - other: int or float or Variable instance
        
        Returns:
            - Variable instance
        '''
        # Sum of derivatives for two Variable types
        try:
            new_f = Variable(self.val + other.val)
            new_f.der = self.der + other.der
        # When other is a real number (beta)
        except AttributeError:
            new_f = Variable(self.val + other)
            new_f.der = self.der
        return new_f
    
    def __neg__(self):
        '''
        Special dunder method to handle the negation of a Variable instance.
        
        Returns:
            - Variable instance
        '''
        return Variable(-1*self.val, -1*self.der)

    def __radd__(self, other):
        '''
        Special dunder method to handle the case of int/float + Variable instance.
        Convert to Variable + int/float and then handled by __add__(self, other).

        Input:
            - other: int or float or Variable instance
        
        Returns:
            - Variable instance
        '''
        return self.__add__(other)

    def __sub__(self, other):
        '''
        Compute the new variable with updated value and derivative after substraction.

        Input:
            - other: int or float or Variable instance
        
        Returns:
            - Variable instance
        '''
        # Subtraction using the dunder methods above
        return self + (-1*other)

    def __rsub__(self, other):
        '''
        Special dunder method to handle the case of int/float - Variable instance.
        Convert to int/float + (-Variable) and then handled by __radd__ and __neg__.

        Input:
            - other: int or float or Variable instance
        
        Returns:
            - Variable instance
        '''
        return other + (-1*self)
    
    def __pow__(self, p):
        '''
        Compute the new variable with updated value and derivative after powering.

        Input:
            - p: int or float
        
        Returns:
            - Variable instance
        '''
        new_f = Variable(self.val ** p)
        new_f.der = p * self.val ** (p - 1) * self.der
        return new_f

    def __lt__(self, other):
        '''
        Comparison operator between Variable instances based on current values.

        Input:
            - other: Variable instance
        
        Returns:
            - boolean, True if self.val < other.val
        '''
        assert type(other) is Variable
        return self.val < other.val 

    def __le__(self, other):
        '''
        Comparison operator between Variable instances based on current values.

        Input:
            - other: Variable instance
        
        Returns:
            - boolean, True if self.val <= other.val
        '''
        assert type(other) is Variable
        return self.val <= other.val 

    def __gt__(self, other):
        '''
        Comparison operator between Variable instances based on current values.

        Input:
            - other: Variable instance
        
        Returns:
            - boolean, True if self.val > other.val
        '''
        assert type(other) is Variable
        return self.val > other.val 

    def __ge__(self, other):
        '''
        Comparison operator between Variable instances based on current values.

        Input:
            - other: Variable instance
        
        Returns:
            - boolean, True if self.val >= other.val
        '''
        assert type(other) is Variable
        return self.val >= other.val 

    def __eq__(self, other):
        '''
        Comparison operator between Variable instances based on current values.

        Input:
            - other: Variable instance
        
        Returns:
            - boolean, True if self.val == other.val
        '''
        assert type(other) is Variable
        return self.val == other.val 

    def __ne__(self, other):
        '''
        Comparison operator between Variable instances based on current values.

        Input:
            - other: Variable instance
        
        Returns:
            - boolean, True if self.val != other.val
        '''
        assert type(other) is Variable
        return self.val != other.val    
        
        
class Variables:
    def __init__(self, n_inputs):
        '''
        Attributes:
            - self.n_inputs: int, number of input variables x, y, z ...
            - self.variables: list of object type Variable
        '''
        self.n_inputs = n_inputs
        self.variables = []
    
    def __len__(self):
        '''
        Returns the number of input variables.

        Returns:
            - int, number of input variables
        '''
        return self.n_inputs
    
    def __iter__(self):
        '''
        Helper method to make a Variables instance iterable.

        Returns:
            - iterable, iterator of Variables
        '''
        return iter(self.variables)
    
    def __getitem__(self, key):
        '''
        Helper method to make a Variables instance subscribable by index.

        Returns:
            - Variable instance
        '''
        assert key < len(self.variables), "Key Error"
        return self.variables[key]
    
    def set_values(self, values):
        '''
        This class is a vector representation of all the input variables.
        Input:
            values: list of float numbers
        Returns:
            None
        '''
        n = len(values)
        assert n == self.n_inputs, 'Dimension Mismatch!'
        variable_list = []
        for i, value in enumerate(values):
            der_list = np.zeros(n)
            der_list[i] = 1
            variable_list.append(Variable(value, der_list))
        self.variables = variable_list
        return self


class Functions():
    def __init__(self, Fs):
        '''
        Input:
            - Fs: List of Variable objects. Each item represent one output dimension.
        '''
        self.Fs = Fs

    def __len__(self):
        return len(self.Fs)
    
    def values(self):
        '''
        Returns the current values of each output

        Returns:
            - ndarray of size len(self.Fs)
        '''
        result = [F.get_value() for F in self.Fs]
        return np.array(result)
    
    def Jacobian(self):
        '''
        Computes the Jacobian matrix.

        Returns:
            - ndarray of shape (n_outputs, n_inputs)
        '''
        return np.vstack([f.get_gradient() for f in self.Fs])


def auto_diff(functions, variable_values):        
    '''
        Differentiate a list of functions in respect to a list of values
        
        Input:
            - functions: A list of python functions to represent vector functions. 
            Each function takes a list of elements to represent variables, and outputs the defined function of those variables.
            - variable_values: A list of integers or floats to represent each variable value.
            
        Returns:
            A tuple which contains an numpy array of each function evaluated at the specified values,
            and the Jacobian of the vector function evaluated at variable values.
    '''

    # Define variables as our variable types
    variables = Variables(n_inputs=len(variable_values))
    variables.set_values(variable_values)
    
    # Apply vector function to vector inputs
    function = Functions(Fs = [f([v for v in variables]) for f in functions])
    
    return function.values(), function.Jacobian()
    


"""
#For some reason this test suite messed with the codecov bad
if __name__ == "__main__":
    from overLoad import *

    variables = Variables(n_inputs=2)
    variables.set_values([3, 1])
    x, y = variables[0], variables[1]

    function = Functions(Fs=[x*y, x ** 2, x * (2+y)]) # 2 inputs, 3 outputs
    print(function.values())
    print()
    print(function.Jacobian()) # 3 by 2 matrix
    
    # We have a new way to differentiate functions
    function1 = lambda v: v[0]*v[1]
    function2 = lambda v: v[0]**2
    function3 = lambda v: v[0]*(2+v[1])
    
    #This returns the same thing as before!!!
    auto_diff([function1, function2, function3], variable_values=[3,1])

"""
