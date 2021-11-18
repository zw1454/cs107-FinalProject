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


if __name__ == '__main__':
    variables = Variables(n=1).set_values(values=[3]) # List of object Variable
    x = variables[0]
    y = (2 * x + x ** 2)
    print(y.der)
    print(y.val)
    print((y ** 2).der)
