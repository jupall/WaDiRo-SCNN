import cvxpy as cp
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from datetime import datetime
import scipy as sc
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



# Define shallow neural network
class SCNN_torch(nn.Module):
    def __init__(self, n_input_layers, n_hidden_layers, n_output_layers, bias) -> None:
        super(SCNN_torch, self).__init__()
        self.entry = nn.Linear(n_input_layers, n_hidden_layers, bias= bias)
        self.act1 = nn.ReLU()
        self.exit = nn.Linear(n_hidden_layers, n_output_layers, bias = bias)


    def forward(self, x):
        out = self.entry(x)
        out = self.act1(out)
        out = self.exit(out)
        return out
    

class scnn():
    """ 
    SCNN with ReLU activation patterns solved with CVXPY. 

    Attributes:
    - bias: The inclusion of bias weights in the model.
    - n_inputs: The number of input features.
    - v_opt: The optimal weigths of the convex formulation of the scnn training.
    - w_opt: The optimal weigths of the convex formulation of the scnn training.
    - b2_opt: The optimal bias of the output neuron.
    - W1_torch: The equivalent weights of the first layer of the scnn model for Pytorch.
    - w2_torch: The equivalent weights of the second layer of the scnn model for Pytorch.
    - b1_torch: The equivalent bias of the first layer of the scnn model for Pytorch.
    - b2_torch: The equivalent bias of the second layer of the scnn model for Pytorch.
    - max_neurons: The maximum number of neurons in the hidden layer.
    - model: The Pytorch equivalent model of the scnn.
    - loss: The loss function to be used ["l1" or "l2"].
    - regularizer: The regularizer to be used ["RIDGE" or "LASSO"].
    - lamb_reg: The regularization parameter.
    - sampled_u: The sampled gate vectors.
    """
    def __init__(self)-> None:
        super(scnn, self).__init__()
        self.bias = None
        self.n_inputs = None
        self.v_opt = None
        self.w_opt = None
        self.b2_opt = None
        self.W1_torch = None
        self.w2_torch = None
        self.b1_torch = None
        self.b2_torch = None
        self.max_neurons = None
        self.model = None
        self.loss = None
        self.regularizer = None
        self.lamb_reg = None
        self.sampled_u = None

        
    def __sample_gate_vectors(self,d:int, n_samples:int, seed:int=42)->np.ndarray:
        rng = np.random.default_rng(seed=seed)
        G = rng.standard_normal((d, int((n_samples+1)/2)))
        
        return G

    def __compute_activation_patterns(
        self,
        X: np.ndarray,
        max_samples: int,
        seed:int=42,
        filter_duplicates: bool = True,
        filter_zero: bool = True): 

        n, d = X.shape
        G = self.__sample_gate_vectors(d, max_samples, seed)
        
        XG = np.matmul(X, G)
        XG = np.maximum(XG, 0)
        XG[XG > 0] = 1
        D = XG

        if filter_duplicates:
            D, indices = np.unique(XG, axis=1, return_index=True)
            G = G[:, indices] # validate 

        # filter out the zero column.
        if filter_zero:
            non_zero_cols = np.logical_not(np.all(D == np.zeros((n, 1)), axis=0))
            D = D[:, non_zero_cols]
            G = G[:, non_zero_cols] # validate
        
        self.sampled_u = G
        return D, G # note: the D matrix, here, is of shape n x max samples, so there are max_samples times vectors that needs to be put in actual matrices with diag()
    
    def __get_equivalent_weights(self):
        
        P, d = self.v_opt.shape
        
        W_1 = np.zeros((2*P,d))
        w_2 = np.zeros((2*P,1))
        
        for i in range(P):
            temp_v = (sc.linalg.norm(self.v_opt[i].reshape(d),2)) # PAS SQUARE ROOT !!
            W_1[i] = self.v_opt[i]/temp_v
            w_2[i] = temp_v
            
            temp_w = (sc.linalg.norm(self.w_opt[i].reshape(d),2)) # pas square root !!
            W_1[i+P] = self.w_opt[i]/temp_w
            w_2[i+P] = -temp_w
            
            
        self.b1_torch = W_1[:,-1] if self.bias else 0 #a check
        self.W1_torch = W_1[:,0:-1] if self.bias else W_1 #a check
        self.w2_torch = w_2
        

        

    def get_torch_model(self, verbose:bool = False)->nn.Module:
        """Generate the equivalent non-convex weigths and introduce them into a Pytorch model.

        Args:
            verbose (bool, optional): Print information on the procedure. Defaults to False.

        Returns:
            nn.Module: a Pytorch neural network.
        """
        P,d = self.v_opt.shape
        if self.bias:
            d = d-1
        model = SCNN_torch(n_input_layers=d, n_hidden_layers=2*P, n_output_layers=1, bias = self.bias)
        model.double()
        
        self.__get_equivalent_weights()
        
        for name, param in model.named_parameters():
            param.data = torch.zeros(param.data.shape)
            if name in 'entry.weight':
                if verbose: print(name, param.data)
                param.data=torch.tensor(self.W1_torch, dtype= torch.float64)
                if verbose: print(param.data)
                
            if name in 'exit.weight':
                if verbose: print(name, param.data)
                param.data = torch.tensor(self.w2_torch.reshape((1,2*P)), dtype= torch.float64)
                if verbose: print(param.data)
                
            if name in 'entry.bias':
                if verbose:  print(name, param.data)
                param.data = torch.tensor(self.b1_torch, dtype= torch.float64)
                if verbose: print(param.data)
                
            if name in 'exit.bias':
                if verbose: print(name, param.data)
                param.data = torch.tensor(self.b2_torch, dtype= torch.float64)
                if verbose: print(param.data)
        self.model = model   
        return model
    
    def train(self, X_train, Y_train, bias:bool, max_neurons:int, loss:str="l1", lamb_reg:float = 0, regularizer:str = "LASSO",solver:str = "CLARABEL", verbose:bool=False):
        """Train the SCNN model to optimality. This training is only optimal if the maximum number of neurons is large enough. 
        Otherwise, the model should still be performant.

        Args:
            X_train (ndartray): Training features.
            Y_train (ndarray): Training labels.
            bias (bool): Inclusion of bias weights.
            max_neurons (int): Maximal number of neurons in the hidden layer.
            loss (str): Loss function to be used ["l1" or "l2"]. Defaults to "l1".
            lamb_reg (float, optional): Regularization parameter. Defaults to 0.
            regularizer (str, optional): Regularization to be used ["RIDGE" or "LASSO"]. Defaults to "LASSO".
            solver (str, optional): Solver to integrate with CVXPY. Defaults to "CLARABEL".
            verbose (bool, optional): Print additional information on solving. Defaults to False.
        Returns:
            float: The value of the optimal objective function.
        """
        N,d = X_train.shape
        self.n_inputs = d
        self.bias = bias
        self.max_neurons = max_neurons
        self.loss = loss
        self.regularizer = regularizer
        self.lamb_reg = lamb_reg
        d_mod = d
        if self.bias:
            X_train = np.hstack((X_train, np.ones((N,1)))) #a tester, hidden neurons bias
            d_mod += 1
            
        D, G = self.__compute_activation_patterns(X_train, max_neurons)
        N,M = D.shape
        


        if bias:
            b2 = cp.Variable(1) #output neuron bias
            v = cp.Variable((M,d_mod))
            w = cp.Variable((M,d_mod))
        else:
            b2 = 0
            v = cp.Variable((M,d))
            w = cp.Variable((M,d))
            
            
        constraints = []

        obj = 0
        big_sum = 0
        for i in range(M):
            Di = np.diag(D[:,i])
        
            A = 2*D[:,i] - np.ones_like(D[:,i])
            B = cp.multiply(A, X_train@v[i])
            C = cp.multiply(A, X_train@w[i])
            constraints += [C >= 0]
            constraints += [B >= 0]

            big_sum += (Di@X_train@(v[i] - w[i])) #there is no regularization for the moment, N
        
        # LOSS FUCNTION
        if loss == "l1":
            b = cp.Variable((N))
            constraints += [big_sum + b2 - Y_train[:,0] <= b]
            constraints += [-(big_sum + b2 - Y_train[:,0]) <= b]
            constraints += [b >=0]
            obj += cp.sum(b)
        elif loss == "l2" or loss != "l1":
            obj += cp.sum_squares(big_sum + b2 - Y_train[:,0])
        
        # REGULARIZER
        if regularizer == "RIDGE" and lamb_reg > 0:
            obj += lamb_reg*cp.sum_squares(v)
            obj += lamb_reg*cp.sum_squares(w)
            #obj += lamb_reg*cp.sum_squares(v-w)
        elif regularizer == "LASSO" and lamb_reg > 0:
            #obj += lamb_reg * cp.mixed_norm(v, p=2, q=1)
            #obj += lamb_reg *cp.mixed_norm(w, p=2, q=1)
            for i in range(M):
                obj += lamb_reg * cp.norm(v[i],1)
                obj += lamb_reg * cp.norm(w[i],1)
                #obj += lamb_reg * cp.norm(v[i]-w[i],1)
        else: pass

        
            
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=solver, verbose = verbose)
        
        self.v_opt=v.value
        self.w_opt = w.value
        self.b2_opt = b2.value if bias else 0
        self.b2_torch = self.b2_opt


        return prob.value
    
    def predict_with_sampled_u(self, x, verbose=False):
        """Predict the output of the SCNN model with the sampled gate vectors. 
        
        This output should be equivalent to the output of the PyTorch model if the number of neurons is large enough.

        Args:
            x (ndarray): Features to predict.
            verbose (bool, optional): Print additional information. Defaults to False.
        """
        N,d = x.shape
        
        d_mod = d
        output_list = []

        if self.bias:
            x = np.hstack((x, np.ones((N,1)))) # hidden neurons bias
            d_mod += 1
       
        G = self.sampled_u.T
        u = np.array(self.v_opt - self.w_opt)
    
        Beta = np.array(u).flatten(order='F').reshape(-1,1) 
        
        for i in range(N):
            d_j = np.array((np.array(G@x[i].T) > 0).reshape(-1,1), dtype = np.float64)

            z = np.array(d_j@x[i,:].reshape((1,d_mod)))
            z = z.flatten(order='F').reshape(-1,1)


            output = np.dot(Beta.T, z) + self.b2_opt
            output_list.append(output)
        return np.array(output_list).flatten().reshape(-1,1)

    
class wadiro_scnn():
    """ 
    SCNN with ReLU activation patterns trained in a tractable order-1 Wasserstein DRO formulation. 

    Attributes:
    - bias: The inclusion of bias weights in the model.
    - n_inputs: The number of input features.
    - v_opt: The optimal weigths of the convex formulation of the scnn training.
    - w_opt: The optimal weigths of the convex formulation of the scnn training.
    - b2_opt: The optimal bias of the output neuron.
    - W1_torch: The equivalent weights of the first layer of the scnn model for Pytorch.
    - w2_torch: The equivalent weights of the second layer of the scnn model for Pytorch.
    - b1_torch: The equivalent bias of the first layer of the scnn model for Pytorch.
    - b2_torch: The equivalent bias of the second layer of the scnn model for Pytorch.
    - max_neurons: The maximum number of neurons in the hidden layer.
    - model: The Pytorch equivalent model of the scnn.
    - radius: The radius of the Wasserstein ball.
    """
    def __init__(self)-> None:
        super(wadiro_scnn, self).__init__()
        self.bias = None
        self.n_inputs = None
        self.v_opt = None
        self.w_opt = None
        self.b2_opt = None
        self.W1_torch = None
        self.w2_torch = None
        self.b1_torch = None
        self.b2_torch = None
        self.max_neurons = None
        self.model = None
        self.radius = None
        self.sampled_u = None
        
    def __sample_gate_vectors(self,d:int, n_samples:int, seed:int=42)->np.ndarray:
        rng = np.random.default_rng(seed=seed)
        G = rng.standard_normal((d, int((n_samples+1)/2)))
        
        return G

    def __compute_activation_patterns(
        self,
        X: np.ndarray,
        max_samples: int,
        seed:int=42,
        filter_duplicates: bool = True,
        filter_zero: bool = True): 

        n, d = X.shape
        G = self.__sample_gate_vectors(d, max_samples, seed)
        
        XG = np.matmul(X, G)
        XG = np.maximum(XG, 0)
        XG[XG > 0] = 1
        D = XG

        if filter_duplicates:
            D, indices = np.unique(XG, axis=1, return_index=True)
            G = G[:, indices]

        # filter out the zero column.
        if filter_zero:
            non_zero_cols = np.logical_not(np.all(D == np.zeros((n, 1)), axis=0))
            D = D[:, non_zero_cols]
            G = G[:, non_zero_cols]

        self.sampled_u = G
        return D, G # note: the D matrix, here, is of shape n x max samples, so there are max_samples times vectors that needs to be put in actual matrices with diag()
    
    def __get_equivalent_weights(self):
        
        P, d = self.v_opt.shape
        
        W_1 = np.zeros((2*P,d))
        w_2 = np.zeros((2*P,1))
        
        for i in range(P):
            if sc.linalg.norm(self.v_opt[i], 2) > 1e-10: # test!!
                #temp_v = np.sqrt(sc.linalg.norm(self.v_opt[i].reshape(d),2))
                temp_v = (sc.linalg.norm(self.v_opt[i].reshape(d),2)) # PAS SQUARE ROOT !!
                W_1[i] = self.v_opt[i]/temp_v
                w_2[i] = temp_v
    
            if sc.linalg.norm(self.w_opt[i], 2) > 1e-10: # test!!
                #temp_w = np.sqrt(sc.linalg.norm(self.w_opt[i].reshape(d),2)) 
                temp_w = (sc.linalg.norm(self.w_opt[i].reshape(d),2))  # pas square root !!
                W_1[i+P] = self.w_opt[i]/temp_w
                w_2[i+P] = -temp_w
            
            
        self.b1_torch = W_1[:,d-1] if self.bias else 0 #a check
        self.W1_torch = W_1[:,0:d-1] if self.bias else W_1 #a check
        self.w2_torch = w_2
        

        

    def get_torch_model(self, verbose:bool=False)->nn.Module:
        """Generate the equivalent non-convex weigths and introduce them into a Pytorch model.

        Args:
            verbose (bool, optional): Print information on the procedure. Defaults to False.

        Returns:
            nn.Module: a Pytorch neural network.
        """
        P,d = self.v_opt.shape
        if self.bias:
            d = d-1
        model = SCNN_torch(n_input_layers=d, n_hidden_layers=2*P, n_output_layers=1, bias = self.bias)
        model.double()
        
        self.__get_equivalent_weights()
        
        for name, param in model.named_parameters():
            param.data = torch.zeros(param.data.shape)
            if name in 'entry.weight':
                if verbose: print(name, param.data)
                param.data=torch.tensor(self.W1_torch, dtype= torch.float64)
                if verbose: print(param.data)
                
            if name in 'exit.weight':
                if verbose: print(name, param.data)
                param.data = torch.tensor(self.w2_torch.reshape((1,2*P)), dtype= torch.float64)
                if verbose: print(param.data)
                
            if name in 'entry.bias':
                if verbose: print(name, param.data)
                param.data = torch.tensor(self.b1_torch, dtype= torch.float64)
                if verbose: print(param.data)
                
            if name in 'exit.bias':
                if verbose: print(name, param.data)
                param.data = torch.tensor(self.b2_torch, dtype= torch.float64)
                if verbose: print(param.data)
        self.model = model   
        return model
    
        

    def train(self, X_train, Y_train, radius:float, bias:bool, max_neurons:int, solver:str = "CLARABEL", verbose:bool=False, wasserstein:str="l1"):
        """Train the WaDiRo-SCNN model to optimality. This training is only optimal if the maximum number of neurons is large enough. 
        Otherwise, the model should still be performant.

        Args:
            X_train (ndartray): Training features.
            Y_train (ndarray): Training labels.
            radius (float): Radius of the Wasserstein ball.
            bias (bool): Inclusion of bias weights.
            max_neurons (int): Maximal number of neurons in the hidden layer.
            solver (str, optional): Solver to integrate with CVXPY. Defaults to "CLARABEL".
            verbose (bool, optional): Print additional information on solving. Defaults to True.
            wasserstein (str, optional): The norm to be used in the definition of the wasserstein distance ["l1" or "l2"]. Defaults to "l1".
        Returns:
            float: The value of the optimal objective function.
        """
        N,d = X_train.shape
        self.n_inputs = d
        self.bias = bias
        self.radius = radius
        d_mod = d
        if self.bias:
            X_train = np.hstack((X_train, np.ones((N,1)))) #a tester, hidden neurons bias
            d_mod += 1
            
        D, G = self.__compute_activation_patterns(X_train, max_neurons)
        N,M = D.shape
        

        if self.bias:
            a = cp.Variable(1) #equivalent to kappa
            c = cp.Variable(N) #absolute in objective
            b2 = cp.Variable(1) #output neuron bias
            v = cp.Variable((M,d_mod))
            w = cp.Variable((M,d_mod))
        else:
            a = cp.Variable(1)
            c = cp.Variable(N)
            b2 = 0
            v = cp.Variable((M,d))
            w = cp.Variable((M,d))
            
            
        constraints = []
        obj = 0
        
        # constraints on original problem
        for i in range(M):    
            A = 2*D[:,i] - np.ones_like(D[:,i]) 
            B = cp.multiply(A, X_train@v[i]) 
            C = cp.multiply(A, X_train@w[i])
            constraints += [C >= 0] # cone constraints
            constraints += [B >= 0] # cone constraints
        
        
        # DRO constraint (Kappa) for l1 wasserstein distance
        if wasserstein == "l1":
            constraints += [b2 <= a] # 
            constraints += [-b2 <= a] # 
            constraints += [v<= a] #equivalent to saying vec(U)_i <= a \forall i in [[P*d]]
            constraints += [-(v)<= a] #equivalent to saying vec(U)_i <= a \forall i in [[P*d]]
            constraints += [-w<= a] #equivalent to saying vec(U)_i <= a \forall i in [[P*d]]
            constraints += [(w)<= a] #equivalent to saying vec(U)_i <= a \forall i in [[P*d]]
            constraints += [a>=1]
            constraints += [c >= 0]
        elif wasserstein == "l2" or wasserstein != "l1":
        # DRO constraint (Kappa) for l2 wassterstein distance 
            constraints += [cp.norm(cp.hstack([cp.vec(v),cp.vec(-w), b2, 1]), 2) <= a]
            constraints += [a >= 0, c>= 0]
            
        
    
        # absolute value on the loss function
        for j in range(N):
            constraints += [D[j]@(v-w)@cp.transpose(X_train[j]) + b2  - Y_train[j,0] <= c[j]]
            constraints += [-D[j]@(v-w)@cp.transpose(X_train[j]) - b2 + Y_train[j,0] <= c[j]]
        
        

        # objective
        obj += cp.sum(c)/N
        obj += radius * a 
        
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=solver, verbose = verbose)
        
        self.v_opt= v.value
        self.w_opt = w.value
        self.b2_opt =  b2.value if self.bias else 0
        self.b2_torch = self.b2_opt

        return prob.value
    
    def predict_with_sampled_u(self, x, verbose=False):
        """Make a prediction with the trained WaDiRo-SCNN model and its sampled gate vectors. 
        
        This output should be equivalent to the output of the PyTorch model if the number of neurons is large enough.

        Args:
            x (ndarray): Features to predict.
            verbose (bool, optional): Print additional information. Defaults to False.
        """
        N,d = x.shape
        
        d_mod = d
        output_list = []

        if self.bias:
            x = np.hstack((x, np.ones((N,1)))) # hidden neurons bias
            d_mod += 1
       
        G = self.sampled_u.T
        u = np.array(self.v_opt - self.w_opt)
    
        Beta = np.array(u).flatten(order='F').reshape(-1,1) 
        
        for i in range(N):
            d_j = np.array((np.array(G@x[i].T) > 0).reshape(-1,1), dtype = np.float64)

            z = np.array(d_j@x[i,:].reshape((1,d_mod)))
            z = z.flatten(order='F').reshape(-1,1)


            output = np.dot(Beta.T, z) + self.b2_opt
            output_list.append(output)
        return np.array(output_list).flatten().reshape(-1,1)
    
class wadiro_linreg():
    """
    A Wasserstein DRO linear regression implementation with CVXPY.

    Attributes:
    - bias: The inclusion of a bias weight in the model.
    - Beta: The optimal weights of the linear regression.
    - b: The optimal bias of the linear regression.
    - n_inputs: The number of input features.
    - radius: The radius of the Wasserstein ball.
    """
    def __init__(self)-> None:
        super(wadiro_linreg, self).__init__()
        self.bias = True # always true
        self.Beta = None # weights
        self.b = None # value of the bias
        self.n_inputs = None
        self.radius = None
    
    def train(self, X_train, Y_train, radius:float, solver:str = "CLARABEL", wasserstein:str='l1', verbose:bool = False):
        """Train the WaDiRo-LinReg model to optimality with CVXPY.

        Args:
            X_train (ndarray): Training features.
            Y_train (ndarray): Training labels.
            radius (float): The radius of the Wasserstein ball.
            solver (str, optional): Solver to integrate with CVXPY. Defaults to "CLARABEL".
            wasserstein (str, optional): The norm to be used in the definition of the wasserstein distance ["l1" or "l2"]. Defaults to 'l1'.
            verbose (bool, optional): Print additional information on solving. Defaults to True.
        """
        self.radius = radius
        self.n_inputs = X_train.shape[1]
        N = X_train.shape[0] # Number of data point
        P = X_train.shape[1] + 1# Number of weight in the regression
        c = cp.Variable(N) # intermediate variable
        a = cp.Variable() # intermediate variable
        Beta = cp.Variable(P) # weight vector

        # Define the problem constraints
        constraints = []

        if wasserstein == "l1":
            constraints += [a>=Beta]
            constraints += [a>=-Beta]
            constraints += [a>=1]
        elif wasserstein == "l2" or wasserstein != "l1":
            constraints += [cp.norm(Beta,2)<=a]
            constraints += [a>=0]
        for i in range(N):
            constraints += [Y_train[i] - ((X_train[i, :] @ Beta[1:P]) + Beta[0]) <= c[i]]
            constraints += [-Y_train[i] + ((X_train[i, :] @ Beta[1:P]) + Beta[0]) <= c[i]]
            constraints += [c[i]>= 0]
        
        # Define the objective function
        obj_dro = a*radius + cp.sum(c)/N

        # Solve the problem by minimizing the objective function, given the constraints
        prob_dro = cp.Problem(cp.Minimize(obj_dro),constraints)
        prob_dro.solve(solver = solver, verbose = verbose)
        
        self.Beta = Beta.value[1:P]
        self.b = Beta.value[0]

        # Display
        if verbose:
            print("The optimal value is:", prob_dro.value, "$")
            print("The optimal Beta is:", Beta.value, "$")
            print("The optimal a is:", a.value, "$")
            print("The optimal c is:", c.value, "$")
        
    def predict(self, X):
        N = X.shape[0]
        predictions = [X[i, :] @ self.Beta + self.b for i in range(N)]
        predictions = np.array(predictions).reshape((N, 1))
        return predictions

        
class linreg():
    """ 
    A classical linear regression with LASSO or RIDGE regularizer.
    """
    def __init__(self)-> None:
        super(linreg, self).__init__()
        self.bias = True # always true
        self.Beta = None # weights
        self.b = None # value of the bias
        self.n_inputs = None
        self.lamb_reg = None
        self.loss = None
        self.regularizer = None
    
    def train(self, X_train, Y_train, loss:str="l1", lamb_reg:float = 0, regularizer:str = "LASSO", solver:str = "CLARABEL", verbose:bool = True):
        self.lamb_reg = lamb_reg
        self.n_inputs = X_train.shape[1]
        self.loss = loss
        self.regularizer = regularizer
        N = X_train.shape[0] # Number of data point
        P = X_train.shape[1] + 1# Number of weight in the regression
        Beta = cp.Variable(P) # weight vector
        

        # Define the problem constraints
        constraints = []

        # LOSS FUNCTION
        if loss == "l1" or loss != "l2":
            c = cp.Variable(N)
            for i in range(N):
                constraints += [Y_train[i] - ((X_train[i, :] @ Beta[1:P]) + Beta[0]) <= c[i]]
                constraints += [-Y_train[i] + ((X_train[i, :] @ Beta[1:P]) + Beta[0]) <= c[i]]
                constraints += [c[i]>= 0]
            # Define the objective function
            obj = cp.sum(c)
        elif loss == "l2":
            obj = cp.sum_squares((Y_train[:,0] - (X_train @ Beta[1:P])) - Beta[0])
            
        # REGULARIZER
        if regularizer == "LASSO" and lamb_reg>0:
            obj += lamb_reg*cp.sum(cp.abs(Beta))
        elif regularizer == "RIDGE" and lamb_reg>0:
            obj += lamb_reg*cp.sum_squares(Beta)
        else:
            pass



        # Solve the problem by minimizing the objective function, given the constraints
        prob = cp.Problem(cp.Minimize(obj),constraints)
        prob.solve(solver = solver, verbose = verbose)
        
        self.Beta = Beta.value[1:P]
        self.b = Beta.value[0]

        # Display
        if verbose:
            print("The optimal value is:", prob.value, "$")
            print("The optimal Beta is:", Beta.value, "$")
        return prob.value

        
    def predict(self, X):
        N = X.shape[0]
        predictions = [X[i, :] @ self.Beta + self.b for i in range(N)]
        predictions = np.array(predictions).reshape((N, 1))
        return predictions
    
        

""" This cutom dataset facilitate the transition between our data and Pytorch's dataloader"""
class CustomDataset(Dataset): # the one to use for multiswag
    def __init__(self, X, y):
        self.y = torch.tensor(y).float()
        self.X = torch.tensor(X).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        x = self.X[i]
        return x, self.y[i]

"""
class FNN_Model(nn.Module):
    def __init__(self, dropout_p, n_input_layers, n_hidden, n_output_layers) -> None:
        super(FNN_Model, self).__init__()
        self.entry = nn.Linear(n_input_layers, n_hidden)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.act4 = nn.ReLU()
        self.exit = nn.Linear(n_hidden, n_output_layers)

    def forward(self, x):
        out = self.entry(x)
        out = self.act1(out)
        out = self.dropout1(out)  # dropout layer
        out = self.fc1(out)
        out = self.act2(out)
        out = self.dropout2(out) 
        out = self.fc2(out)
        out = self.act3(out)
        out = self.dropout3(out)
        out = self.fc3(out)
        out = self.act4(out)
        out = self.exit(out)
        return out
    
# function to train the model with the training dataloader
def train_FNN_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train() # model is ready to be trained, ex: dropout is active
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    for X, y in data_loader:
        optimizer.zero_grad() # precaution
        X = X.to(device)
        y = y.to(device)
        output = model.forward(X)
        loss = loss_function(output, y)
        loss.backward() # train
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss 
    #print(f"Train loss: {avg_loss}")

# function to test the model with the validation dataloader
def test_FNN_model(data_loader, model, loss_function):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    num_batches = len(data_loader)
    total_loss = 0

    model.eval() # eval phase, ex: dropout is inactive
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    return avg_loss

# function to predict outputs with the trained model
def predict_FNN(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    
    return output
"""