import numpy as np

class Variable:
    def __init__(self, value, derivatives=None) -> None:
        '''
        Stores the current value and derivative of this variable.
            - self.val: value
            - self.der: derivative
        '''
        self.val = value
        self.der = derivatives

    def __str__(self):
        return f"Value {self.val}\n" + \
            f"Full Jacobian {self.der}\n"
    
    def get_value(self):
        return self.val
    
    def get_gradient(self):
        return self.der
    
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
    
    # When we use the "-" operator dunder
    def __neg__(self):
        return Variable(-1*self.val, -1*self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Subtraction using the dunder methods above
        return self + (-1*other)

    def __rsub__(self, other):
        return other + (-1*self)
    
    def __pow__(self, p):
        new_f = Variable(self.val ** p)
        new_f.der = p * self.val ** (p - 1) * self.der
        return new_f

    def __lt__(self, other):
        return self.val < other.val 

    def __le__(self, other):
        return self.val <= other.val 

    def __gt__(self, other):
        return self.val > other.val 

    def __ge__(self, other):
        return self.val >= other.val 

    def __eq__(self, other):
        return self.val == other.val 

    def __ne__(self, other):
        return self.val != other.val    
        
        
class Variables:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.variables = []
    
    def __len__(self):
        return self.n_inputs
    
    def __iter__(self):
        return iter(self.variables)
    
    def __getitem__(self, key):
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


class Function():
    def __init__(self, Fs):
        '''
        Input:
            - Fs: List of Variable objects. Each item represent one output dimension.
        '''
        self.Fs = Fs
    
    def values(self):
        result = [F.get_value() for F in self.Fs]
        return np.array(result)
    
    def Jacobian(self):
        n_inputs = len(self.Fs[0].der)
        n_outputs = len(self.Fs)
        result = np.zeros(n_outputs, n_inputs)
        pass



if __name__ == "__main__":
    from overLoad import *

    variables = Variables(n_inputs=2)
    variables.set_values([3, 1])
    x, y = variables[0], variables[1]

    function = Function(Fs=[x*y, x ** 2, x * sin(y)])
    print(function.values())
