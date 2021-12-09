from dualNumbers import *
import numpy as np

def gradient_decent(function, init_variables, learning_rate = 0.1, max_iter = 1000, tol = 0.0001):
    """
    Function that optimizes a python function using gradient decsent.
  
    Inputs:
      - function: A python function that takes a list of elements to represent variables,
                and outputs the defined function of those variables.
      - init_variables: A list of values to evaluate the function at initially.
      - max_iter: The max iterations the algorithm can run. Default to 1000.
      - learning_rate: The learning rate of the gradient decscent algorith. Default set to 0.1.
      - tol: Tolerence used for convergence criteria. When the function evaluation changes
          by less than this tolerance the function finishes. Default set to 0.0001.
          
    Outputs:
      - A tuple of the optimal value and the and the inputs to the function that yielded
        the value
    """
  
    # initialize 
    value, jacobian = auto_diff([function], init_variables)
    curr_w = np.array(init_variables)
    i = 0
    diff = 1
  
    while i<max_iter or diff>tol:
    
        #calc change in weights
        delta_w = -learning_rate * jacobian
    
        #update weigths
        curr_w = curr_w + delta_w
    
        #update function for new values
        last_value = value
        value, jacobian = auto_diff([function], curr_w)
    
        #check for convergence or max tol
        i += 1
        diff = np.abs(value - last_value)
    
    return value, curr_w

def momentum_gd(function, init_variables, momentum = 0.8, learning_rate = 0.1, max_iter = 1000, tol = 0.0001):
    """
    Function that optimizes a python function using gradient descent and momentum.
  
    Inputs:
    - function: A python function that takes a list of elements to represent variables,
                and outputs the defined function of those variables.
    - init_variables: A list of values to evaluate the function at initially.
    - momentum: term to stabilize learning toward the global minimum. Must be set [0,1].
                Default set to 0.9.
    - max_iter: The max iterations the algorithm can run. Default to 1000.
    - learning_rate: The learning rate of the gradient decscent algorith. Default set to 0.1.
    - tol: Tolerence used for convergence criteria. When the function evaluation changes
          by less than this tolerance the function finishes. Default set to 0.0001.
          
    Outputs:
    - A tuple of the optimal value and the and the inputs to the function that yielded
      the value
    """
    #Check to ensure momentum is between 0 and 1
    if momentum < 0 or momentum > 1:
        raise ValueError("Please ensure momentum factor is [0,1].")
    
    #initialize 
    value, jacobian = auto_diff([function], init_variables)
    curr_w = np.array(init_variables)
    i = 0
    diff = 1
    delta_w = np.zeros(len(init_variables))
    
    while i<max_iter or diff>tol:
        #calc change in weights
        delta_w = -learning_rate*jacobian + momentum * delta_w
    
        #update weigths
        curr_w = curr_w + delta_w
    
        #update function for new values
        last_value = value
        value, jacobian = auto_diff([function], curr_w)
    
        #check for convergence or max tol
        i += 1
        diff = np.abs(value - last_value)
    
    return value, curr_w

def adagrad(function, init_variables, learning_rate = 0.1, epsilon=1e-8, max_iter = 1000):
    """
    Function that optimizes a python function via the adagrad algorithm. 
  
    Inputs:
      - function: A python function that takes a list of elements to represent variables,
                and outputs the defined function of those variables.
      - init_variables: A list of values to evaluate the function at initially.
      - max_iter: The max iterations the algorithm can run.
      - learning_rate: The learning rate of the gradient decscent algorithm.
      - epsilon: smoothing term that avoids division by zero. Should be resonably small.
                default set to 1e-8
      
      Outputs:
      - A tuple of the optimal value and the and the inputs to the function that yielded
      the value
     """
    # First, obtain the partial derivative of the objective function w.r.t. to the value parameter 
    # Repeat this within the while loop 
    value, jacobian = auto_diff([function], init_variables)
    curr_w = np.array(init_variables)
    epsilon = epsilon
    
    # We then need to calculate the square of the partial derivative of each 
    # variable and add them to the running sum of these values.
    gradientsum = 0 

    for i in range(max_iter):
        #calc delta
        gradientsum = gradientsum + jacobian**2
        delta_var = (learning_rate * jacobian) / np.sqrt(gradientsum + epsilon)
        
        #update weights
        curr_w = curr_w - delta_var 
        
        #take a step
        last_value = value
        value, jacobian = auto_diff([function], curr_w)

    return value, curr_w

if __name__ == "__main__":
    function = lambda v: v[0]**2
  
    print(gradient_decent(function, [1]))
    
    print(momentum_gd(function, [1]))
    
    print(adagrad(function, [1]))
  
