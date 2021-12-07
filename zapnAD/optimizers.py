from dualNumbers import *
import numpy as np

def gradient_decent(function, init_variables, max_iter = 1000, learning_rate = 0.1, tol = .0001):
  """
  Function that optimizes a python function.
  
  Inputs:
    - function: A python function that takes a list of elements to represent variables,
                and outputs the defined function of those variables.
    - init_variables: A list of values to evaluate the function at initially.
    - max_iter: The max iterations the algorithm can run.
    - learning_rate: The learning rate of the gradient decscent algorith.
    - tol: Tolerence used for convergence criteria. When the function evaluation changes
          by less than this tolerance the function finishes.
          
  Outputs:
    - A tuple of the optimal value and the and the inputs to the function that yielded
      the value
  """
  
  #initialize 
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

def adagrad(function, init_variables, max_iter=1000, learning_rate = 0.1, epsilon=1e-8):
  """
  Function that optimizes a python function via the adagrad algorithm. 
  
  Inputs:
    - function: A python function that takes a list of elements to represent variables,
                and outputs the defined function of those variables.
    - init_variables: A list of values to evaluate the function at initially.
    - max_iter: The max iterations the algorithm can run.
    - learning_rate: The learning rate of the gradient decscent algorithm.
    - epsilon: smoothing term that avoids division by zero
          
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
    gradientsum = gradientsum + jacobian**2
    delta_var = (learning_rate * jacobian) / np.sqrt(gradientsum + epsilon)
    current_values = curr_w - delta_var 
    value, jacobian = auto_diff([function], current_values)

  return value, current_values

if __name__ == "__main__":
  function = lambda v: v[0]**2
  
  print(gradient_decent(function, [1]))
  
