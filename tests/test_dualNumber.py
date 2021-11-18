import pytest
import numpy as np
from zapnAD.dualNumbers import *

class TestVariable:
    
    @classmethod
    def setup_class(TestVariable):
        """Set up variables to use in many test cases."""
        v1 = Variable(5, 1.5)
        v2 = Variable(4.0, 2)
        return [v1, v2]
        
    
    def test_one(self):
        """ Test default initialization of variable"""
        v = Variable(5)
        assert v.val == 5
        assert v.der == 1
        
    def test_two(self):
        """ Tests initialization with setting the derivative"""
        v = Variable(5, der=0)
        assert v.der == 0

    def test_three(self):
        """ Tests initializing with value = 0"""
        v = Variable(0)
               
    def test_four(self):
        """Test the __str__ function."""
        v = Variable(5)
        print(str(v))
        assert str(v) == "Dual Number: Value 5, Derivative: 1."
        
    def test_five(self):
        """Test multiplication with two dual numbers."""
        v1, v2 = self.setup_class()
        res = v1*v2
        
        assert res.val == 20
        assert res.der == 16
        
    def test_six(self):
        """Test __mul__: multiplication with a real with real on the 
        RIGHT hand side of the *."""
        v1, v2 = self.setup_class()
        res = v1*6
        
        assert res.val == 30
        assert res.der == 9
        
    def test_seven(self):
        """Test __rmul__ multiplication with a real, with real on the
        LEFT hand side of the *."""
        v1, v2 = self.setup_class()
        res = 6.0*v1
        
        assert res.val == 30
        assert res.der == 9
        
    def test_eight(self):
        """Test addition of two dual numbers."""
        v1, v2 = self.setup_class()
        res = v1 + v2
        
        assert res.val == 9
        assert res.der == 3.5
        
    def test_nine(self):
        """Test __add__ with real, with real on RIGHT side of the +."""
        v1, v2 = self.setup_class()
        res = v1 + 2.0
        
        assert res.val ==7
        assert res.der == 1.5
        
    def test_ten(self):
        """Test __radd__ with real, with real on LEFT side of the +."""
        v1, v2 = self.setup_class()
        res = 2.0 + v1
        
        assert res.val == 7
        assert res.der == 1.5
        
    def test_eleven(self):
        """Test __neg__ """
        v1 = Variable(5, 1.5)
        v2 = Variable(-4, -1)
        
        res1 = -v1
        assert res1.val == -5
        assert res1.der == -1.5
        
        res2 = -v2
        assert res2.val == 4
        assert res2.der == 1
        
    def test_twelve(self):
        """Test subtraction with two dual numbers."""
        v1, v2 = self.setup_class()
        res = v1-v2
        
        assert res.val == 1
        assert res.der == -0.5
        
    def test_thirteen(self):
        """Test subtraction with real on the right hand side."""
        v1, v2 = self.setup_class()
        res = v1-3
        
        assert res.val == 2
        assert res.der == 1.5
        
    def test_fourteen(self):
        """Test __rsub__: subtraction with real on left hand side."""
        v1, v2 = self.setup_class()
        res = 3-v1
        
        assert res.val == -2
        assert res.der == -1.5
        
    def test_fifteen(self):
        """Test __pow__ with positive power"""
        v1, v2 = self.setup_class()
        res = v2**3
        
        assert res.val == 64
        assert res.der == 96
        
    def test_sixteen(self):
        """Test __pow__ with negative power"""
        
        v1, v2 = self.setup_class()
        res = v2**-3
        
        assert res.val == 1/64.0
        assert res.der == -3.0/128
        
    def test_seventeen(self):
        """Test __pow__ with zero."""
        v1, v2 = self.setup_class()
        res = v2**0
        
        assert res.val == 1
        assert res.der == 0
        
class TestVariables:
   
    def test_one(self):
        """Test creation of variables."""
        vs = Variables(4)
        
        assert len(vs) == 4
   
   
    def test_two(self):
        """Test setting value of variables with correct value, list format."""
        v_const = Variables(4)
        v_objs = v_const.set_values([5, 5, 5, 5])
        
        for v in v_objs:
            assert v.val == 5
            assert v.der == 1
            
    def test_three(self):
        """Test setting value of variables with correct value, numpy array format."""
        v_const = Variables(4)
        v_objs = v_const.set_values(np.array([5, 5, 5, 5]))
        
        for v in v_objs:
            assert v.val == 5
            assert v.der == 1
            
    def test_four(self):
        """Test setting value of variables with incorrect value."""
        v_const = Variables(4)
        
        with pytest.raises(AssertionError):
            v_const.set_values([5, 5])
        
        
    
"""
TODO:

TestVariable test_three
TestVariable test_fifteen check
"""
