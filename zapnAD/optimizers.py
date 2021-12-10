from .dualNumbers import *
import numpy as np

__all__ = ['Optimizer', 'GradientDescentOptimizer', 'MomentumOptimizer', 'AdaGradOptimizer', 'AdamOptimizer']

class Optimizer():
    """Class representing an optimizer of a python function."""
    
    def __init__(self,  learning_rate=0.1, max_iter = 1000, tol = 1e-8):
        
        """
          Initializes the optimizer parameters
  
          Arguments:
          - max_iter: The max iterations the algorithm can run. Default to 1000.
          - learning_rate: The learning rate of the gradient decscent algorith. Default set to 0.1.
          - tol: Tolerence used for convergence criteria. When the function evaluation changes
                 by less than this tolerance the function finishes. Default set to 1e-8
          
        """

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = 1e-8
        self.delta_ws = []
        self.prev_values = []
        self.prev_jacobians = []
        

    def _step(self):
        """The calculation done at each step of the optimizer"""
        raise NotImplementedError
        
        
    def optimize(self, function, init_variables):
        """Optimizes the given function.
        
        Arguments:
        - function: A python function that takes a list of elements to represent variables,
                and outputs the defined function of those variables.
        - init_variables: A list of values to evaluate the function at initially.
        
        Returns:
        value at optimum
        """

        curr_w = np.array(init_variables)
        array_shape = curr_w.shape
        self.delta_ws.append(np.zeros(array_shape))

        val, der = auto_diff([function], curr_w)
        self.prev_values.append(val)
        self.prev_jacobians.append(der)
        
        self.i = 0
        self.diff  = 1
        
        while self.i < self.max_iter and self.diff > self.tol:
            
            delta_w = self._step().reshape(array_shape)
            self.delta_ws.append(delta_w)
            
            curr_w = curr_w + delta_w
            
            val, der = auto_diff([function], curr_w)
            
            self.i += 1
            self.diff = np.abs(val - self.prev_values[-1])
            
            self.prev_values.append(val)
            self.prev_jacobians.append(der)
            
        return val, curr_w
    
    def get_values(self):
        """Returns array of the function value at each step, size n_steps x 1"""
        return np.vstack(self.prev_values)
    
    def get_jacobians(self):
        """Returns array of the gradient at each step, size n_steps x n_variables"""
        return np.vstack(self.prev_jacobians)
    
    def get_step_deltas(self):
        """Returns array of the step size at each step, size n_steps x n_variables"""
        return np.vstack(self.delta_ws)
    
class GradientDescentOptimizer(Optimizer):
    
    def __init__(self,  learning_rate=0.1, max_iter = 1000, tol=1e-8):
        """Initializes parameters for the gradient descent optimizer"""
        
        super().__init__(learning_rate=learning_rate, max_iter = max_iter, tol=tol)
        
    def _step(self):
        """Defines the delta in independent variable values at a given step for gradient descent."""
        #calc change in weights
        return -self.learning_rate * self.prev_jacobians[-1]
    
class MomentumOptimizer(Optimizer):
    
    def __init__(self, momentum=0.8, learning_rate=0.1, max_iter = 1000, tol=1e-8):  
        """Initializes parameters for the Momentum optimizer
        
        Arguments:
        - momentum: term to stabilize learning toward the global minimum. Must be set [0,1].
                Default set to 0.9.
        """ 
        
        super().__init__(learning_rate=learning_rate, max_iter = max_iter, tol=tol)
        self.momentum = momentum
        
    def _step(self):
        """Defines the delta in independent variable values at a given step for the momentum optimizer."""
        
        #calc change in weights
        return -self.learning_rate*self.prev_jacobians[-1] + self.momentum * self.delta_ws[-1]
        
        
class AdamOptimizer(Optimizer):
    
    def __init__(self, b_1=0.9, b_2=0.999, error=10e-8, learning_rate=0.1, max_iter = 1000, tol=1e-8):
        """Initializes parameters for the Adam optimizer
        
        Arguments:
        - b_1: ADAM optimizer hyperparameter controlling first moment term
        - b_2: ADAM optimizer hyperparameter controlling second moment term
        - error: ADAM optimizer hyperparameter preventing division by 0
        """ 
       
        super().__init__(learning_rate=learning_rate, max_iter = max_iter, tol=tol)
        self.m, self.v, self.m_corr, self.v_corr = 0, 0, 0, 0
        self.b_1 = b_1
        self.b_2 = b_2
        self.error = error
        
    def _step(self):
        """Defines the delta in independent variable values at a given step for the Adam optimizer."""
        
        self.m = self.b_1*self.m + (1-self.b_1)*self.prev_jacobians[-1]
        self.m_corr = self.m/(1-np.power(self.b_1, self.i+1))

        
        self.v = self.b_2*self.v + (1-self.b_2)*self.prev_jacobians[-1]**2
        self.v_corr = self.v/(1-np.power(self.b_2, self.i+1))

        # update derivative
        return -np.array(self.learning_rate*(self.m_corr/(np.sqrt(self.v_corr)+self.error)))
    
class AdaGradOptimizer(Optimizer):
    
    def __init__(self, epsilon = 1e-8,  learning_rate=0.1, max_iter = 1000, tol=1e-8):
        """Initializes parameters for the AdaGrad optimizer
        
        Arguments:
        - epsilon: smoothing term that avoids division by zero. Should be resonably small.
                default set to 1e-8
        """
        super().__init__(learning_rate=learning_rate, max_iter = max_iter, tol=tol)
        self.epsilon = epsilon
        self.gradientsum = 0
        
    def _step(self):
        """Defines the delta in independent variable values at a given step for the AdaGrad optimizer."""

        #calc delta
        self.gradientsum = self.gradientsum + self.prev_jacobians[-1]**2
        return -(self.learning_rate * self.prev_jacobians[-1]) / np.sqrt(self.gradientsum + self.epsilon)
        

