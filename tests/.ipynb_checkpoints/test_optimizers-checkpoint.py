import pytest
import numpy as np
from zapnAD.optimizers import *

class TestOptimizers():
    
    @classmethod
    def setup_class(TestOptimizers):
        """Set up variables to use in many test cases."""
        f1 = lambda v: v[0]**2 
        f2 = lambda v: v[0]**2 + v[1]**3 + 2
        return [f1, f2]
    
    def test_one_a(self):
        """test gradient decsent for function 1"""
        f1, f2 = self.setup_class()
        r1, r2 = gradient_decent(f1, [1])
        assert r1 == pytest.approx(0, .001)
        assert r2 == pytest.approx(0, .001)
        
    def test_two_a(self):
        """test momentum for function 1"""
        f1, f2 = self.setup_class()
        r1, r2 = momentum_gd(f1, [1])
        assert r1 == pytest.approx(0, .001)
        assert r2 == pytest.approx(0, .001)
        
    def test_three_a(self):
        """test momentum for function 1"""
        f1, f2 = self.setup_class()
        r1, r2 = adagrad(f1, [1])
        assert r1 == pytest.approx(0, .001)
        assert r2 == pytest.approx(0, .001)
