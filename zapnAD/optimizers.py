from dualNumbers import *
import numpy as np

def gradient_decent(function, init_variables, learning_rate = 0.1, max_iter = 1000,  tol = 1e-8):
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
    array_shape = curr_w.shape
    i = 0
    diff = 1
  
    while i<max_iter and diff>tol:
    
        #calc change in weights
        delta_w = -learning_rate * jacobian
    
        #update weigths
        curr_w = curr_w + delta_w.reshape(array_shape)

        
        last_value = value
        value, jacobian = auto_diff([function], curr_w)
    
        #check for convergence or max tol
        i += 1
        diff = np.abs(value - last_value)
    
    return value, curr_w

def momentum_gd(function, init_variables, momentum = 0.8, learning_rate = 0.1, max_iter = 1000, tol = 1e-8):
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
    array_shape = curr_w.shape
    i = 0
    diff = 1
    delta_w = np.zeros(array_shape)
    
    while i<max_iter and diff>tol:
        #calc change in weights
        delta_w = -learning_rate*jacobian + momentum * delta_w
    
        #update weigths
        delta_w = delta_w.reshape(array_shape)
        curr_w = curr_w + delta_w
    
        #update function for new values
        last_value = value
        
        value, jacobian = auto_diff([function], curr_w)
    
        #check for convergence or max tol
        i += 1
        diff = np.abs(value - last_value)
    
    return value, curr_w


def adam(function, init_variables, max_iter = 1000,  tol = 1e-8, b_1=0.9, b_2=0.999, error=10e-8, learning_rate=0.01):
    """
    Function that optimizes a python function using Adaptive Moment Estimation (Adam).
    
    Inputs:
    - function: A python function that takes a list of elements to represent variables,
                and outputs the defined function of those variables.
    - init_variables: A list of values to evaluate the function at initially.
    - max_iter: The max iterations the algorithm can run.
    - learning_rate: The learning rate of the gradient decscent algorith.
    - tol: Tolerence used for convergence criteria. When the function evaluation changes
          by less than this tolerance the function finishes.
    - b_1: ADAM optimizer hyperparameter controlling first moment term
    - b_2: ADAM optimizer hyperparameter controlling second moment term
    - error: ADAM optimizer hyperparameter preventing division by 0
 
    Outputs:
    - A tuple of the optimal value and the and the inputs to the function that yielded
      the value
    """

    #initialize 
    val, der = auto_diff([function], init_variables)

    curr_w = np.array(init_variables)
    array_shape = curr_w.shape
    diff = 1
    
    m, v, m_corr, v_corr = 0, 0, 0, 0
    i = 0
    
    while i<max_iter and diff>tol:
        
        m = b_1*m + (1-b_1)*der
        m_corr = m/(1-np.power(b_1, (i+1)))

        
        v = b_2*v + (1-b_2)*der**2
        v_corr = v/(1-np.power(b_2, i+1))

        # update derivative
        delta_w = np.array(learning_rate*(m_corr/(np.sqrt(v_corr)+error))).reshape(array_shape)
        curr_w = curr_w - delta_w

        prev_val = val
        val, der = auto_diff([function], curr_w)

        i += 1
        diff = np.abs(val - prev_val)


    return val, curr_w

                                       
def adagrad(function, init_variables, learning_rate = 0.1, epsilon=1e-8, max_iter = 1000, tol = 1e-8):
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
      - tol: Tolerence used for convergence criteria. When the function evaluation changes
          by less than this tolerance the function finishes.
      
      Outputs:
      - A tuple of the optimal value and the and the inputs to the function that yielded
      the value
     """
    # First, obtain the partial derivative of the objective function w.r.t. to the value parameter 
    # Repeat this within the while loop 
    value, jacobian = auto_diff([function], init_variables)
    curr_w = np.array(init_variables)
    array_shape = curr_w.shape
    epsilon = epsilon
    
    # We then need to calculate the square of the partial derivative of each 
    # variable and add them to the running sum of these values.
    gradientsum = 0 
    
    # stopping variables
    diff = 1
    i = 0

    while i<max_iter and diff>tol: 
        #calc delta
        gradientsum = gradientsum + jacobian**2
        delta_var = (learning_rate * jacobian) / np.sqrt(gradientsum + epsilon)
        delta_var = delta_var.reshape(array_shape)
        
        #update weights
        curr_w = curr_w - delta_var 
            
        prev_value = value
        value, jacobian = auto_diff([function], curr_w)
        
        i += 1
        diff = np.abs(value - prev_value)

    return value, curr_w

if __name__ == "__main__":
    pass