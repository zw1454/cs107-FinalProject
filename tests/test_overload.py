import pytest
import numpy as np
from dualNumbers import *
from overLoad import * 

class TestVariable:
    
    @classmethod
    def setup_class(TestVariable):
        """Set up variables to use in many test cases."""
        v1 = Variable(5, 1.5)
        v2 = 4 # randomly choose a real number to test with the overloaded functions. 
        return [v1, v2]

    def test_one(self):
        """ Test sin function"""
        v1, v2 = self.setup_class()
        assert sin(v1) == Variable(np.sin(v1.val), np.cos(v1.val) * v1.der)
        assert sin(v2) == np.sin(4)

    def test_two(self):
        """ Test cos function"""
        v1, v2 = self.setup_class()
        assert cos(v1) == Variable(np.cos(v1.val), np.cos(v1.val) * v1.der)
        assert cos(v2) == np.cos(4)

    def test_three(self):
        """ Test tan function"""
        v1, v2 = self.setup_class()
        assert tan(v1) == Variable(np.tan(v1.val), np.tan(v1.val) * v1.der)
        assert tan(v2) == np.tan(4)

    def test_four(self):
        """ Test arcsin function"""
        v1, v2 = self.setup_class()
        assert arcsin(v1) == Variable(np.arcsin(v1.val), np.cos(v1.val) * v1.der)
        assert arcsin(v2) == np.arcsin(4)

    def test_five(self):
        """ Test arccos function"""
        v1, v2 = self.setup_class()
        assert arccos(v1) == Variable(np.arccos(v1.val), np.arccos(v1.val) * v1.der)
        assert arccos(v2) == np.arccos(4)

    def test_six(self):
        """ Test arctan function"""
        v1, v2 = self.setup_class()
        assert arctan(v1) == Variable(np.arctan(v1.val), np.arctan(v1.val) * v1.der)
        assert arctan(v2) == np.arctan(4)

    def test_seven(self):
        """ Test exponent function"""
        v1, v2 = self.setup_class()
        assert exp(v1) == Variable(np.exp(v1.val), np.exp(v1.val) * v1.der)
        assert exp(v2) == np.exp(4)

    def test_eight(self):
        """ Test log base exp(1) function"""
        v1, v2 = self.setup_class()
        assert np.log(v1) == Variable(np.log(v1.val), (1/v1.val) * v1.der)
        assert log(v2) == np.log(4)

    def test_nine(self):
        """ Test log base 2 function"""
        v1, v2 = self.setup_class()
        assert log2(v1) == Variable(np.log2(v1.val), (1/(v1.val * np.log(2))) * v1.der)
        assert log2(v2) == np.log2(4)

    def test_ten(self):
        """ Test log base 10 function"""
        v1, v2 = self.setup_class()
        assert log10(v1) == Variable(np.log10(v1.val), (1/(v1.val * np.log(2))) * v1.der)
        assert log10(v2) == np.log10(4)

    def test_eleven(self):
        """ Test square root function"""
        v1, v2 = self.setup_class()
        assert sqrt(v1) == Variable(v1.val * 0.5, 0.5 * v1.val ** (1/2 - 1) * v1.der)
        assert sqrt(v2) == np.sqrt(4)