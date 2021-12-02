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
    def __init__(self, n):
        self.n = n
    
    def __len__(self):
        return self.n
    
    def set_values(self, values):
        '''
        This class is a vector representation of all the input variables.
        Input:
            values: list of float numbers
        Returns:
            A list of single variables of user-specified length
        '''
        n = len(values)
        assert n == self.n, 'Dimension Mismatch!'
        variable_list = []
        for i, value in enumerate(values):
            der_list = np.zeros(n)
            der_list[i] = 1
            variable_list.append(Variable(value, der_list))
        return variable_list


if __name__ == "__main__":
    variables = Variables(n=2).set_values([1, 2])
    x, y = variables[0], variables[1]
    print((x + y) * x)