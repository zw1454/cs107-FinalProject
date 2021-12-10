import pytest
import numpy as np
from zapnAD.dualNumbers import *
from zapnAD.overLoad import * 

class TestVariable:
    
    @classmethod
    def setup_class(TestVariable):
        """Set up variables to use in many test cases."""
        v1 = Variable(5, np.array([1.5]))
        v2 = 4 # randomly choose a real number to test with the overloaded functions. 
        return [v1, v2]

    def test_one(self):
        """ Test sin function"""
        v1, v2 = self.setup_class()
        assert sin(v1).val == np.sin(5) 
        assert sin(v1).der == np.array([np.cos(5)*1.5])
        assert sin(v2) == np.sin(4)

    def test_two(self):
        """ Test cos function"""
        v1, v2 = self.setup_class()
        assert cos(v1).val == np.cos(5)
        assert cos(v1).der == np.array([-np.sin(5)*1.5])
        assert cos(v2) == np.cos(4)

    def test_three(self):
        """ Test tan function"""
        v1, v2 = self.setup_class()
        assert tan(v1).val == np.tan(5)
        assert tan(v1).der == np.array([1.5/(np.cos(5)**2)])
        assert tan(v2) == np.tan(4)
        
    def test_four_a(self):
        """ Test arcsin function within domain"""
        v1 = Variable(0.8, np.array([0.5])) # test in domain of function
        v2 = -.8 # test in domain of function
        assert arcsin(v1).val == np.arcsin(0.8)
        assert arcsin(v1).der == np.array([0.5/np.sqrt(1-0.8**2)])
        assert arcsin(v2) == np.arcsin(-0.8)

    def test_four_b(self):
        """ Test arcsin function at edge of domain (infinite slope)"""
        v1 = Variable(1, np.array([0.5])) # test in domain of function
        v2 = -.8 # test in domain of function
        assert arcsin(v1).val == np.arcsin(1)
        assert arcsin(v1).der == np.array([float('Inf')])
        assert arcsin(v2) == np.arcsin(-0.8)
        
    def test_five_a(self):
        """ Test arccos function within domain."""
        v1 = Variable(0.8, np.array([0.5])) # test in domain of function
        v2 = -.8 # test in domain of function
        assert arccos(v1).val == np.arccos(0.8)
        assert arccos(v1).der == np.array([-0.5/np.sqrt(1-0.8**2)])
        assert arccos(v2) == np.arccos(-0.8)
        
    def test_five_b(self):
        """ Test arccos function at edge of domain (infinite slope)"""
        v1 = Variable(1, np.array([0.5])) # test in domain of function
        v2 = -.8 # test in domain of function
        assert arccos(v1).val == np.arccos(1)
        assert arccos(v1).der == np.array([-float('Inf')])

    def test_six(self):
        """ Test arctan function"""
        v1, v2 = self.setup_class()
        assert arctan(v1).val == np.arctan(5)
        assert arctan(v1).der == np.array([1.5/(1+5**2)])
        assert arctan(v2) == np.arctan(4)

    def test_seven(self):
        """ Test exponent function"""
        v1, v2 = self.setup_class()
        assert exp(v1).val == np.exp(5)
        assert pytest.approx(exp(v1).der[0], 1e-7) == 1.5*np.exp(5)
        assert exp(v2) == np.exp(4)

    def test_eight(self):
        """ Test log base exp(1) function"""
        v1, v2 = self.setup_class()
        assert log(v1).val == np.log(5)
        assert pytest.approx(log(v1).der[0], 1e-7) == 1.5/5.0
        assert log(v2) == np.log(4)

    def test_nine(self):
        """ Test log base 2 function"""
        v1, v2 = self.setup_class()
        assert log2(v1).val == np.log2(5)
        assert pytest.approx(log2(v1).der[0], 1e-7) == 1.5/(np.log(2)*5)
        assert log2(v2) == np.log2(4)

    def test_ten(self):
        """ Test log base 10 function"""
        v1, v2 = self.setup_class()
        assert log10(v1).val == np.log10(5)
        assert log10(v1).der == np.array([1.5/(np.log(10)*5)])
        assert log10(v2) == np.log10(4)

    def test_eleven(self):
        """ Test square root function"""
        v1, v2 = self.setup_class()
        assert sqrt(v1).val == np.sqrt(5)
        assert sqrt(v1).der == np.array([1.5*(1/2)*5**(-1/2)])
        assert sqrt(v2) == np.sqrt(4)

    def test_eleven_b(self):
        """ Test square root of negative number"""
        v1 = Variable(-1, np.array([0.5])) # test in domain of function
        with pytest.raises(ValueError, match=r"Value < 0 not valid for square root"):
            sqrt(v1)
    
    def test_twelve(self):
        """Test sinh"""
        v1, v2 = self.setup_class()
        assert sinh(v1).val == np.sinh(5)
        assert sinh(v2) == np.sinh(4)
        
    def test_thirteen(self):
        """Test cosh"""
        v1, v2 = self.setup_class()
        assert cosh(v1).val == np.cosh(5)
        assert cosh(v2) == np.cosh(4)
        
    def test_fourteen(self):
        """Test tanh"""
        v1, v2 = self.setup_class()
        assert tanh(v1).val == np.tanh(5)
        assert tanh(v2) == np.tanh(4)
    
    def test_fifteen_a(self):
        """Test Log With Different Bases"""
        v1, v2 = self.setup_class()
        assert log(v1, 2) == log2(v1)
        assert log(v2, 2) == log2(v2)
        
    def test_fifteen_b(self):
        """Test to make sure we arent doing a Log to a negative number"""
        v1, v2 = self.setup_class()
        neg = -2
        with pytest.raises(ValueError, match="Log base must be greater than or equal to 1"):
            log(v1, neg)
        
    def test_sixteen(self):
        """Test expinitials to different Bases"""
        v1, v2 = self.setup_class()
        base = 2
        assert exp(v1,base).val == base**5
        assert pytest.approx(exp(v1,base).der[0], 1e-7) == (5*base**(5-1))*1.5
        assert exp(v2, base) == base**4

    