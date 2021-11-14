from typing import List

class Variable:
    def __init__(self, value) -> None:
        self.val = value
        self.der = 1
    
    def __mul__(self, other):
        # Product derivative rule for two function types
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
        # Sum of derivatives for two function types
        try:
            new_f = Variable(self.val + other.val)
            new_f.der = self.der + other.der
        # When other is a real number (beta)
        except AttributeError:
            new_f = Variable(self.val + other)
            new_f.der = self.der
        return new_f
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __pow__(self, p):
        new_f = Variable(self.val ** p)
        new_f.der = p * self.val ** (p - 1) * self.der
        return new_f


class Variables:
    def __init__(self, n) -> None:
        self.n = n
    
    def set_values(self, values: List[float]) -> List[Variable]:
        assert len(values) == self.n, 'Dimension Mismatch!'
        return [Variable(value) for value in values]


if __name__ == '__main__':
    x = Variables(n=1).set_values(values=[3])[0]
    y = (x + x ** 2)
    print(y.der)
    print(y.val)
    print((y ** 2).der)