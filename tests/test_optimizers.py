import pytest
import numpy as np
from zapnAD.optimizers import *

class TestOptimizers():
    
    @classmethod
    def setup_class(TestOptimizers):
        """Set up variables to use in many test cases."""
        f1 = lambda v: v[0]**2 #univariate case
        f2 = lambda v: v[0]**2 + v[1]**2 #multi-variate case
        return [f1, f2]
    
    def test_one_a(self):
        """test gradient decsent for function 1"""
        print("test 1a")
        f1, f2 = self.setup_class()
        r1, r2 = gradient_decent(f1, [1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        
    def test_one_b(self):
        """test gradient decsent for function 2"""
        print("test 1b")
        f1, f2 = self.setup_class()
        r1, r2 = gradient_decent(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)
        
    def test_two_a(self):
        """test momentum for function 1"""
        print("test 2a")
        f1, f2 = self.setup_class()
        r1, r2 = momentum_gd(f1, [1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        
    def test_two_b(self):
        """test momentum for function 2"""
        print("test 2b")
        f1, f2 = self.setup_class()
        r1, r2 = momentum_gd(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)
        
    def test_three_a(self):
        """test adagrad for function 1"""
        print("test 3a")
        f1, f2 = self.setup_class()
        r1, r2 = adagrad(f1, [1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)

    def test_three_b(self):
        """test adagrad for function 2"""
        print("test 3b")
        f1, f2 = self.setup_class()
        r1, r2 = adagrad(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)
        
    def test_four_a(self):
        """test adam optimizer for function 1"""
        print("test 4a")
        f1, f2 = self.setup_class()
        r1, r2 = adam(f1, [1])
        assert r1[0] ==pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)

    def test_four_b(self):
        """test adam optimizer for function 2"""
        print("test 4b")
        f1, f2 = self.setup_class()
        r1, r2 = adam(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)

