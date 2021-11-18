import pytest
import numpy as np
from dualNumbers import *
from 

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
        assert 