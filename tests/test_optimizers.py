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
    
    def test_pre_optimizer(self):
        f1, f2 = self.setup_class()
        opt = Optimizer()
        with pytest.raises(NotImplementedError):
            opt.optimize(f1, [1])
    
    def test_one_a(self):
        """test gradient decsent for function 1"""
        print("test 1a")
        f1, f2 = self.setup_class()
        opt = GradientDescentOptimizer()
        r1, r2 = opt.optimize(f1, [1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        
    def test_one_b(self):
        """test gradient decsent for function 2"""
        print("test 1b")
        f1, f2 = self.setup_class()
        opt = GradientDescentOptimizer()
        r1, r2 = opt.optimize(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)
        
    def test_one_c(self):
        """test gradient decsent for function 2 w/ step limit"""
        print("test 1a")
        f1, f2 = self.setup_class()
        opt = GradientDescentOptimizer(max_iter=100, learning_rate=0.05, tol=1e-9)
        r1, r2 = opt.optimize(f2, [1, 1])
        assert opt.get_values().shape[0] <= 101
        assert opt.get_jacobians().shape[0] <= 101
        assert opt.get_jacobians().shape[1] == 2
        assert opt.get_step_deltas().shape[0] <= 101
        assert opt.get_step_deltas().shape[1] == 2

        
    def test_two_a(self):
        """test momentum for function 1"""
        print("test 2a")
        f1, f2 = self.setup_class()
        mo = MomentumOptimizer()
        r1, r2 = mo.optimize(f1, [1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        
    def test_two_b(self):
        """test momentum for function 2"""
        print("test 2b")
        f1, f2 = self.setup_class()
        mo = MomentumOptimizer()
        r1, r2 = mo.optimize(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)
    
    def test_two_c(self):
        """test momentum for function 2 w/ step limit"""
        print("test 1a")
        f1, f2 = self.setup_class()
        opt = MomentumOptimizer(max_iter=100, learning_rate=0.05, tol=1e-9)
        r1, r2 = opt.optimize(f2, [1, 1])
        assert opt.get_values().shape[0] <= 101
        assert opt.get_jacobians().shape[0] <= 101
        assert opt.get_jacobians().shape[1] == 2
        assert opt.get_step_deltas().shape[0] <= 101
        assert opt.get_step_deltas().shape[1] == 2
        
    def test_three_a(self):
        """test adagrad for function 1"""
        print("test 3a")
        f1, f2 = self.setup_class()
        ad = AdaGradOptimizer()
        r1, r2 = ad.optimize(f1, [1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)

    def test_three_b(self):
        """test adagrad for function 2"""
        print("test 3b")
        f1, f2 = self.setup_class()
        ad = AdaGradOptimizer()
        r1, r2 = ad.optimize(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)
        
    def test_three_c(self):
        """test adagrad for function 2 w/ step limit"""
        print("test 1a")
        f1, f2 = self.setup_class()
        opt = AdaGradOptimizer(max_iter=100, learning_rate=0.05, tol=1e-9)
        r1, r2 = opt.optimize(f2, [1, 1])
        assert opt.get_values().shape[0] <= 101
        assert opt.get_jacobians().shape[0] <= 101
        assert opt.get_jacobians().shape[1] == 2
        assert opt.get_step_deltas().shape[0] <= 101
        assert opt.get_step_deltas().shape[1] == 2
        
    def test_four_a(self):
        """test adam optimizer for function 1"""
        print("test 4a")
        f1, f2 = self.setup_class()
        adam = AdamOptimizer()
        r1, r2 = adam.optimize(f1, [1])
        assert r1[0] ==pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)

    def test_four_b(self):
        """test adam optimizer for function 2"""
        print("test 4b")
        f1, f2 = self.setup_class()
        adam = AdamOptimizer()
        r1, r2 = adam.optimize(f2, [1,1])
        assert r1[0] == pytest.approx(0, abs=0.001)
        assert r2[0] == pytest.approx(0, abs=0.001)
        assert r2[1] == pytest.approx(0, abs=0.001)
        
    def test_four_c(self):
        """test adam for function 2 w/ step limit"""
        print("test 1a")
        f1, f2 = self.setup_class()
        opt = AdamOptimizer(max_iter=100, learning_rate=0.05, tol=1e-9)
        r1, r2 = opt.optimize(f2, [1, 1])
        assert opt.get_values().shape[0] <= 101
        assert opt.get_jacobians().shape[0] <= 101
        assert opt.get_jacobians().shape[1] == 2
        assert opt.get_step_deltas().shape[0] <= 101
        assert opt.get_step_deltas().shape[1] == 2

