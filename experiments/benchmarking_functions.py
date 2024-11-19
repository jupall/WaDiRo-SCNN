
import sys
sys.path.append('../src/wadiroscnn')
import numpy as np
import scipy as sc
import models as hm
import benchmark_functions as bf
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from datetime import datetime
import torch.nn as nn
import cvxpy as cp
import mlflow.pytorch
from mlflow.models.signature import infer_signature, set_signature
from functools import partial
import sliced_wasserstein_utils as sw
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import dfs, as_apply
from hyperopt.pyll.stochastic import implicit_stochastic_symbols
from hyperopt import hp, fmin, tpe, anneal, Trials, STATUS_OK,  pyll
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, make_scorer
from functools import partial



n_dim = 10
FUNCTIONS_LIST = [ 
	[bf.Hypersphere, n_dim, "Hypersphere"], 
	[bf.Hyperellipsoid, n_dim, "Hyperellipsoid"], 
	[bf.Ackley, n_dim, "Ackley"],
	[bf.Rosenbrock, n_dim, "Rosenbrock"],
	[bf.Rastrigin,n_dim, "Rastrigin"],
	[bf.Schwefel,n_dim, "Schwefel"],
	[bf.Griewank,n_dim, "Griewank"],
	[bf.Ackley,n_dim, "Ackley"],
	[bf.Michalewicz,n_dim, "Michalewicz"],
	[bf.EggHolder,n_dim, "EggHolder"], 				
	[bf.Keane,n_dim, "Keane"],
	[bf.Rana,n_dim, "Rana"], 			
	[bf.Easom,2, "Easom"], # 2 dims
	[bf.DeJong5,2, "DeJong5"], # 2 dims
	[bf.GoldsteinAndPrice,2, "GoldsteinAndPrice"],  # 2 dims
	[bf.PichenyGoldsteinAndPrice,2, "PichenyGoldsteinAndPrice"],
	[bf.StyblinskiTang,n_dim, "StyblinskiTang"],
	[bf.McCormick, 2, "McCormick"], # 2 dims
	[bf.MartinGaddy,2, "MartinGaddy"],
	[bf.Schaffer2, 2, "Schaffer2"], # 2 dims
	[bf.Himmelblau, 2, "Himmelblau"],# 2 dims
	[bf.PitsAndHoles,n_dim, "PitsAndHoles"] 
	]


"""_summary_
Class that simplifies data generation and train-val-test splitting for benchmarking functions.
It generates a dataset with the function, does the scaling and is able to corrupt both inputs and outputs.
TO DO: Option to add a gaussian noise on training inputs.
"""

class dataset_wrapper():
    def __init__(self, benchmark_func, n_dim, rng, N = 1000, train_size = 0.6, test_size=0.5)-> None:
        super(dataset_wrapper, self).__init__()
        self.X = None # always true
        self.Y = None # weights
        self.X_train_scaled = None
        self.Y_train_scaled = None
        self.X_val_scaled = None
        self.Y_val_scaled= None
        self.X_test_scaled = None
        self.Y_test_scaled= None
        self.bench_func = benchmark_func(n_dim) if n_dim !=2 else benchmark_func() 
        self.n_dim = n_dim
        self.scaler_x = StandardScaler(with_mean=False, with_std=False)
        self.scaler_y = StandardScaler(with_mean=False, with_std=False)
        self.rng = rng
        self.N = N
        self.test_size = test_size
        self.train_size = train_size

    def generate_data(self, corrupt_inputs_prob:float = 0, corrupt_outputs_prob:float=0, corrupt_std:float= 4):
        lower_bounds = self.bench_func.suggested_bounds()[0]
        upper_bounds = self.bench_func.suggested_bounds()[1]

        self.X = self.rng.uniform(low = lower_bounds[0], high=upper_bounds[0], size = (self.N, self.n_dim))
        self.Y = np.array([self.bench_func(self.X[i]) for i in range(self.N)])

        # train test split for training and validation phase
        X_train, X_val, Y_train, Y_val = train_test_split(self.X, self.Y, train_size=self.train_size, shuffle=False)

        X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val,test_size= self.test_size, shuffle = False )

        mean_x = np.mean(X_train, axis = 0)
        std_x = np.std(X_train, axis = 0)

        mean_y = np.mean(Y_train)
        std_y = np.std(Y_train)
        # Scale
        self.scaler_x = self.scaler_x.fit(X_train)  # fit scaler on training inputs to avoid data leakage
        self.X_train_scaled = self.scaler_x.transform(X_train)  # transform
        self.X_val_scaled = self.scaler_x.transform(X_val)  # transform
        self.X_test_scaled = self.scaler_x.transform(X_test)  # transform

        Y_train = np.reshape(Y_train, (Y_train.size, 1))
        self.scaler_y = self.scaler_y.fit(Y_train)  # fit scaler on training outputs to avoid data leakage
        self.Y_train_scaled = self.scaler_y.transform(Y_train)  # transform
        self.Y_val_scaled = self.scaler_y.transform(np.reshape(Y_val, (Y_val.size, 1)))  # transform
        self.Y_test_scaled = self.scaler_y.transform(np.reshape(Y_test, (Y_test.size, 1)))  # transform

        if corrupt_inputs_prob > 0:
            #corrupting some data
            for d in range(self.X_train_scaled.shape[1]):
                mask = self.rng.uniform(0,1,size=self.X_train_scaled[:,d].shape) >= 1 - corrupt_inputs_prob
                self.X_train_scaled[:,d][mask] = self.X_train_scaled[:,d][mask] + corrupt_std*self.rng.choice([-std_x[d],std_x[d]], size = self.X_train_scaled[:,d][mask].shape) # noise is a number of stds outside

        if corrupt_outputs_prob > 0: 
            mask = self.rng.uniform(0,1,size=self.Y_train_scaled.shape) >= 1 - corrupt_outputs_prob
            self.Y_train_scaled[mask] =  self.Y_train_scaled[mask] +  corrupt_std*self.rng.choice([-std_y,std_y], size = self.Y_train_scaled[mask].shape) # noise is a number of stds outside
        return True
   

    def generate_data_wasserstein(self, corrupt_data_points:int = 0, min_distance:float = 0.2, n_projections:int = 100, k_multiplier:int = 1.1, L:float=2, seed:int = 42):
        lower_bounds = self.bench_func.suggested_bounds()[0]
        upper_bounds = self.bench_func.suggested_bounds()[1]
        

        self.X = self.rng.uniform(low = lower_bounds[0], high=upper_bounds[0], size = (self.N, self.n_dim))
        self.Y = np.array([self.bench_func(self.X[i]) for i in range(self.N)])

        # train test split for training and validation phase
        X_train, X_val, Y_train, Y_val = train_test_split(self.X, self.Y, train_size=self.train_size, shuffle=False)

        X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val,test_size= self.test_size, shuffle = False )

        # Scale
        self.scaler_x = self.scaler_x.fit(X_train)  # fit scaler on training inputs to avoid data leakage
        self.X_train_scaled = self.scaler_x.transform(X_train) + self.rng.normal(0, 0.05, X_train.shape)  # transform
        self.X_val_scaled = self.scaler_x.transform(X_val) + self.rng.normal(0, 0.05, X_val.shape)# transform
        self.X_test_scaled = self.scaler_x.transform(X_test)  # transform

        Y_train = np.reshape(Y_train, (Y_train.size, 1))
        self.scaler_y = self.scaler_y.fit(Y_train)  # fit scaler on training outputs to avoid data leakage
        self.Y_train_scaled = self.scaler_y.transform(Y_train) + self.rng.normal(0, 0.05, Y_train.shape) # transform
        self.Y_val_scaled = self.scaler_y.transform(np.reshape(Y_val, (Y_val.size, 1))) + self.rng.normal(0, 0.05, Y_val.shape) # transform
        self.Y_test_scaled = self.scaler_y.transform(np.reshape(Y_test, (Y_test.size, 1)))  # transform

        
        if corrupt_data_points > 0: 
            train_signal = np.hstack((self.X_train_scaled, self.Y_train_scaled))
            train_signal_sliced = sw.sliced_wasserstein_outlier_introducer(train_signal, [lower_bounds, upper_bounds], self.bench_func, self.n_dim, corrupt_data_points, min_distance, n_projections, k_multiplier, L, seed, self.rng)
            
            self.X_train_scaled = train_signal_sliced[:,:-1]
            self.Y_train_scaled = np.reshape(train_signal_sliced[:,-1], (train_signal_sliced[:,-1].size, 1))

        return True
    
    def generate_data_wasserstein_corrupt_both(self, corrupt_data_points:int = 0, min_distance:float = 0.2, n_projections:int = 100, k_multiplier:int = 1.1, L:float=2, seed:int = 42):
        lower_bounds = self.bench_func.suggested_bounds()[0]
        upper_bounds = self.bench_func.suggested_bounds()[1]
        
        self.X = self.rng.uniform(low = lower_bounds[0], high=upper_bounds[0], size = (self.N, self.n_dim))
        self.Y = np.array([self.bench_func(self.X[i]) for i in range(self.N)])
       

        # train test split for training and validation phase
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_size, shuffle=False)
        Y_train = np.reshape(Y_train, (Y_train.size, 1))
        Y_test = np.reshape(Y_test, (Y_test.size, 1))
        # TODO: Apply the outlier introducer on both training and validation
        if corrupt_data_points > 0: 
            train_signal = np.hstack((X_train, Y_train))
            train_signal_sliced = sw.sliced_wasserstein_outlier_introducer(train_signal, [lower_bounds, upper_bounds], self.bench_func, self.n_dim, corrupt_data_points, min_distance, n_projections, k_multiplier, L, seed, self.rng)
            
            X_train = train_signal_sliced[:,:-1]
            Y_train = np.reshape(train_signal_sliced[:,-1], (train_signal_sliced[:,-1].size, 1))


        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size= self.train_size, shuffle = True)

        # Scale
        self.scaler_x = self.scaler_x.fit(X_train)  # fit scaler on training inputs to avoid data leakage
        self.X_train_scaled = self.scaler_x.transform(X_train) + self.rng.normal(0, 0.05, X_train.shape)  # transform
        self.X_val_scaled = self.scaler_x.transform(X_val) + self.rng.normal(0, 0.05, X_val.shape)# transform
        self.X_test_scaled = self.scaler_x.transform(X_test)  # transform

        Y_train = np.reshape(Y_train, (Y_train.size, 1))
        self.scaler_y = self.scaler_y.fit(Y_train)  # fit scaler on training outputs to avoid data leakage
        self.Y_train_scaled = self.scaler_y.transform(Y_train) + self.rng.normal(0, 0.05, Y_train.shape) # transform
        self.Y_val_scaled = self.scaler_y.transform(np.reshape(Y_val, (Y_val.size, 1))) + self.rng.normal(0, 0.05, Y_val.shape) # transform
        self.Y_test_scaled = self.scaler_y.transform(np.reshape(Y_test, (Y_test.size, 1)))  # transform

        return True

def test_torch_model(model, X, y, criterion):
    
    total_loss = 0
    
    output = model(torch.tensor(X, dtype=torch.float64))
    total_loss += criterion(output, torch.tensor(y, dtype=torch.float64))

    return total_loss, output       

def test_lin_reg(model, X,y, criterion):
    total_loss = 0
    
    output = model.predict(X)
    total_loss += criterion( torch.tensor(output, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
    
    return total_loss, output
    
        
def objective_DR_SCNN(params, data, criterion, solver_name, experiment, n_corrupted_points, func_name, verbose):
    # unwrap hyperparameters:
    radius = float(params['radius'])
    max_neurons = int(params['max_neurons'])
    bias = True #bool(params['bias'])
    wasserstein = str(params['wasserstein'])

    # print info on run
    print("------start of trial: ------")
    print(f"radius= {radius}, max_neurons = {max_neurons}, bias = {bias}")


    start_time_model = datetime.now() # start timer for whole training
    # define model
    model = hm.wadiro_scnn()
    
    
    with mlflow.start_run() as run:
       
       
        # train
        model.train(X_train=data.X_train_scaled, Y_train=data.Y_train_scaled, radius = radius, bias = bias, max_neurons=max_neurons, verbose=verbose, solver=solver_name, wasserstein=wasserstein)
        
        # torch model
        model_torch = model.get_torch_model(verbose=verbose)
        
        # get training loss 
        train_loss, output = test_torch_model(model_torch, data.X_train_scaled, data.Y_train_scaled, criterion=criterion)
        
        # get validation loss
        val_loss, output =  test_torch_model(model_torch, data.X_val_scaled, data.Y_val_scaled, criterion=criterion)
        
        # get test loss
        test_loss, output = test_torch_model(model_torch,data.X_test_scaled, data.Y_test_scaled, criterion=criterion )

        # get Mean squared error
        mse = nn.MSELoss()
        train_MSE, output = test_torch_model(model_torch, data.X_train_scaled, data.Y_train_scaled, criterion=mse)
        
        # get validation loss
        val_MSE, output =  test_torch_model(model_torch, data.X_val_scaled, data.Y_val_scaled, criterion=mse)
        
        # get test loss
        test_MSE, output = test_torch_model(model_torch,data.X_test_scaled, data.Y_test_scaled, criterion=mse )

        end_time_model =  datetime.now() 
    
        mlflow.set_tag("model_name", "DR_SCNN")
        mlflow.log_param("radius", radius)
        mlflow.log_param("max_neurons", max_neurons)
        mlflow.log_param("bias", bias)
        mlflow.log_param("benchmark_functions", func_name)
        mlflow.log_param("data", data)
        mlflow.log_param("solver", solver_name)
        mlflow.log_param("n_corrupted_points", n_corrupted_points)
        mlflow.log_param("wasserstein distance", wasserstein)


        # Start training and testing

        mlflow.log_metric("MAE_val", val_loss.detach().numpy())
        mlflow.log_metric("MAE_train", train_loss.detach().numpy())
        mlflow.log_metric("MAE_test", test_loss.detach().numpy())
        mlflow.log_metric("MSE_val", val_MSE.detach().numpy())
        mlflow.log_metric("MSE_train", train_MSE.detach().numpy())
        mlflow.log_metric("MSE_test", test_MSE.detach().numpy())
        #signature = infer_signature(np.array(data.X_val_scaled, dtype=np.float64), np.array(model_torch.forward(torch.Tensor(data.X_val_scaled, dtype=torch.float64)), dtype =np.float64))
        mlflow.pytorch.log_model(pytorch_model=model_torch, artifact_path=experiment.artifact_location)

        mlflow.log_param("training time", end_time_model - start_time_model)
        print(f'Training duration: {end_time_model - start_time_model}')

        # save model
        now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")
        str_path = f"./logged_models/DR_SCNN_{now_string}.pt"
        mlflow.set_tag("path_to_model", str_path)
        
        mlflow.pytorch.save_state_dict(
                model_torch.state_dict(),
                path=str_path,
            )
        mlflow.end_run() #end run
        
    return  {
         "status": STATUS_OK,
         "loss": val_loss
        }


def objective_SCNN(params, data, criterion, solver_name, experiment, n_corrupted_points, func_name, verbose):
    # unwrap hyperparameters:
    #function_name = params['function_name']
    max_neurons = int(params['max_neurons'])
    bias = True #bool(params['bias'])
    loss = "l1" #str(params['loss'])
    regularizer = str(params['regularizer'])
    lamb_reg = float(params['lamb_reg'])


    
    
    # print info on run
    print("------start of trial: ------")
    print(f"lamb_reg= {lamb_reg}, max_neurons = {max_neurons}, bias = {bias}")
    
    # make datasets 


    start_time_model = datetime.now() # start timer for whole training
    # define model
    model = hm.scnn()
    
    
    with mlflow.start_run() as run:
       
        # train
        model.train(X_train=data.X_train_scaled, Y_train=data.Y_train_scaled, lamb_reg=  lamb_reg, bias = bias, max_neurons=max_neurons, verbose=verbose, solver=solver_name, loss=loss, regularizer=regularizer)
        
        # torch model
        model_torch = model.get_torch_model(verbose=verbose)
        
        # get training loss 
        train_loss, output = test_torch_model(model_torch, data.X_train_scaled, data.Y_train_scaled, criterion=criterion)
        
        # get validation loss
        val_loss, output =  test_torch_model(model_torch, data.X_val_scaled, data.Y_val_scaled, criterion=criterion)
        
        # get test loss
        test_loss, output = test_torch_model(model_torch,data.X_test_scaled, data.Y_test_scaled, criterion=criterion )

        mse = nn.MSELoss()
        train_MSE, output = test_torch_model(model_torch, data.X_train_scaled, data.Y_train_scaled, criterion=mse)
        
        # get validation loss
        val_MSE, output =  test_torch_model(model_torch, data.X_val_scaled, data.Y_val_scaled, criterion=mse)
        
        # get test loss
        test_MSE, output = test_torch_model(model_torch,data.X_test_scaled, data.Y_test_scaled, criterion=mse )

        end_time_model =  datetime.now() 
    
        mlflow.set_tag("model_name", "SCNN")
        mlflow.log_param("regularizer", regularizer)
        mlflow.log_param("loss", loss)
        mlflow.log_param("lamb_reg", lamb_reg)
        mlflow.log_param("max_neurons", max_neurons)
        mlflow.log_param("bias", bias)
        mlflow.log_param("benchmark_functions", func_name)
        mlflow.log_param("data", data)
        mlflow.log_param("solver", solver_name)
        mlflow.log_param("n_corrupted_points", n_corrupted_points)
        #mlflow.log_param("output_noise_prob",output_noise_prob)


        # Start training and testing

        mlflow.log_metric("MAE_val", val_loss.detach().numpy())
        mlflow.log_metric("MAE_train", train_loss.detach().numpy())
        mlflow.log_metric("MAE_test", test_loss.detach().numpy())
        mlflow.log_metric("MSE_val", val_MSE.detach().numpy())
        mlflow.log_metric("MSE_train", train_MSE.detach().numpy())
        mlflow.log_metric("MSE_test", test_MSE.detach().numpy())
        #signature = infer_signature(np.array(data.X_val_scaled, dtype=np.float64), np.array(model_torch.forward(torch.Tensor(data.X_val_scaled, dtype=torch.float64)), dtype =np.float64))
        mlflow.pytorch.log_model(pytorch_model=model_torch, artifact_path=experiment.artifact_location)

        mlflow.log_param("training time", end_time_model - start_time_model)
        print(f'Training duration: {end_time_model - start_time_model}')

        # save model
        now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")
        str_path = f"./logged_models/SCNN_{now_string}.pt"
        mlflow.set_tag("path_to_model", str_path)
        
        mlflow.pytorch.save_state_dict(
                model_torch.state_dict(),
                path=str_path,
            )
        mlflow.end_run() #end run
        
    return  {
         "status": STATUS_OK,
         "loss": val_loss
        }

def objective_SCNN_no_reg(params, data, criterion, solver_name, experiment, n_corrupted_points, func_name, verbose):
    # unwrap hyperparameters:
    #function_name = params['function_name']
    max_neurons = int(params['max_neurons'])
    bias = True #bool(params['bias'])
    loss = "l1" #str(params['loss'])
    lamb_reg = 0


    
    
    # print info on run
    print("------start of trial: ------")
    print(f"lamb_reg= {lamb_reg}, max_neurons = {max_neurons}, bias = {bias}")
    
    # make datasets 


    start_time_model = datetime.now() # start timer for whole training
    # define model
    model = hm.scnn()
    
    
    with mlflow.start_run() as run:
       
        # train
        model.train(X_train=data.X_train_scaled, Y_train=data.Y_train_scaled, lamb_reg=  lamb_reg, bias = bias, max_neurons=max_neurons, verbose=verbose, solver=solver_name, loss=loss)
        
        # torch model
        model_torch = model.get_torch_model(verbose=verbose)
        
        # get training loss 
        train_loss, output = test_torch_model(model_torch, data.X_train_scaled, data.Y_train_scaled, criterion=criterion)
        
        # get validation loss
        val_loss, output =  test_torch_model(model_torch, data.X_val_scaled, data.Y_val_scaled, criterion=criterion)
        
        # get test loss
        test_loss, output = test_torch_model(model_torch,data.X_test_scaled, data.Y_test_scaled, criterion=criterion )

        mse = nn.MSELoss()
        train_MSE, output = test_torch_model(model_torch, data.X_train_scaled, data.Y_train_scaled, criterion=mse)
        
        # get validation loss
        val_MSE, output =  test_torch_model(model_torch, data.X_val_scaled, data.Y_val_scaled, criterion=mse)
        
        # get test loss
        test_MSE, output = test_torch_model(model_torch,data.X_test_scaled, data.Y_test_scaled, criterion=mse )

        end_time_model =  datetime.now() 
    
        mlflow.set_tag("model_name", "SCNN_no_reg")
        mlflow.log_param("loss", loss)
        mlflow.log_param("lamb_reg", lamb_reg)
        mlflow.log_param("max_neurons", max_neurons)
        mlflow.log_param("bias", bias)
        mlflow.log_param("benchmark_functions", func_name)
        mlflow.log_param("data", data)
        mlflow.log_param("solver", solver_name)
        mlflow.log_param("n_corrupted_points", n_corrupted_points)
        #mlflow.log_param("output_noise_prob",output_noise_prob)


        # Start training and testing

        mlflow.log_metric("MAE_val", val_loss.detach().numpy())
        mlflow.log_metric("MAE_train", train_loss.detach().numpy())
        mlflow.log_metric("MAE_test", test_loss.detach().numpy())
        mlflow.log_metric("MSE_val", val_MSE.detach().numpy())
        mlflow.log_metric("MSE_train", train_MSE.detach().numpy())
        mlflow.log_metric("MSE_test", test_MSE.detach().numpy())
        #signature = infer_signature(np.array(data.X_val_scaled, dtype=np.float64), np.array(model_torch.forward(torch.Tensor(data.X_val_scaled, dtype=torch.float64)), dtype =np.float64))
        mlflow.pytorch.log_model(pytorch_model=model_torch, artifact_path=experiment.artifact_location)

        mlflow.log_param("training time", end_time_model - start_time_model)
        print(f'Training duration: {end_time_model - start_time_model}')

        # save model
        now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")
        str_path = f"./logged_models/SCNN_no_reg_{now_string}.pt"
        mlflow.set_tag("path_to_model", str_path)
        
        mlflow.pytorch.save_state_dict(
                model_torch.state_dict(),
                path=str_path,
            )
        mlflow.end_run() #end run
        
    return  {
         "status": STATUS_OK,
         "loss": val_loss
        }


def objective_linreg(params, data, criterion, solver_name, experiment, n_corrupted_points, func_name, verbose):
    # unwrap hyperparameters:
    #function_name = params['function_name']
    loss = "l1" #str(params['loss'])
    regularizer = str(params['regularizer'])
    lamb_reg = float(params['lamb_reg'])


    
    
    # print info on run
    print("------start of trial: ------")
    print(f"lamb_reg= {lamb_reg} bias = {True}")
    
    # make datasets 

    start_time_model = datetime.now() # start timer for whole training
    # define model
    model = hm.linreg()
    
    
    with mlflow.start_run() as run:
       
        # train
        model.train(X_train=data.X_train_scaled, Y_train=data.Y_train_scaled, lamb_reg=  lamb_reg, verbose=verbose, solver=solver_name, loss=loss, regularizer=regularizer)
        
       
        # get training loss 
        train_loss, output = test_lin_reg(model, data.X_train_scaled, data.Y_train_scaled, criterion=criterion)
        
        # get validation loss
        val_loss, output =  test_lin_reg(model, data.X_val_scaled, data.Y_val_scaled, criterion=criterion)
        
        # get test loss
        test_loss, output = test_lin_reg(model, data.X_test_scaled, data.Y_test_scaled, criterion=criterion )

        mse = nn.MSELoss()
        train_MSE, output = test_lin_reg(model, data.X_train_scaled, data.Y_train_scaled, criterion=mse)
        
        # get validation loss
        val_MSE, output =  test_lin_reg(model, data.X_val_scaled, data.Y_val_scaled, criterion=mse)
        
        # get test loss
        test_MSE, output = test_lin_reg(model,data.X_test_scaled, data.Y_test_scaled, criterion=mse )

        end_time_model =  datetime.now() 
    
        mlflow.set_tag("model_name", "linreg")
        mlflow.log_param("lamb_reg", lamb_reg)
        mlflow.log_param("bias", True)
        mlflow.log_param("benchmark_functions", func_name)
        mlflow.log_param("data", data)
        mlflow.log_param("solver", solver_name)
        mlflow.log_param("n_corrupted_points", n_corrupted_points)
        #mlflow.log_param("output_noise_prob",output_noise_prob)
        mlflow.log_param("Beta", model.Beta)
        mlflow.log_param("b", model.b)


        # Start training and testing

        mlflow.log_metric("MAE_val", val_loss)
        mlflow.log_metric("MAE_train", train_loss)
        mlflow.log_metric("MAE_test", test_loss)
        mlflow.log_metric("MSE_val", val_MSE)
        mlflow.log_metric("MSE_train", train_MSE)
        mlflow.log_metric("MSE_test", test_MSE)
        #signature = infer_signature(np.array(data.X_val_scaled, dtype=np.float64), np.array(model_torch.forward(torch.Tensor(data.X_val_scaled, dtype=torch.float64)), dtype =np.float64))
        #mlflow.pytorch.log_model(pytorch_model=model_torch, artifact_path=experiment.artifact_location)

        mlflow.log_param("training time", end_time_model - start_time_model)
        print(f'Training duration: {end_time_model - start_time_model}')

        # save model
        now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")
        str_path = f"/home/julien/Documents/code/maitrise/my version/logged_models/linreg_{now_string}.csv"
        np.savetxt(
                str_path,
                np.concatenate(( model.Beta, np.array([model.b]))),
                delimiter=",",
            )
        mlflow.set_tag("path_to_model", str_path)
        
        mlflow.end_run() #end run
        
    return  {
         "status": STATUS_OK,
         "loss": val_loss
        }
    
    
def objective_DR_linreg(params, data, criterion, solver_name, experiment, n_corrupted_points, func_name, verbose):
    # unwrap hyperparameters:
    #function_name = params['function_name']
    #bias = bool(params['bias'])
    radius = float(params['radius'])
    wasserstein = str(params['wasserstein'])



    
    
    # print info on run
    print("------start of trial: ------")
    print(f"radius= {radius} bias = {True}")
    
    # make datasets 

    start_time_model = datetime.now() # start timer for whole training
    # define model
    model = hm.wadiro_linreg()
    
    
    with mlflow.start_run() as run:
       
        # train
        model.train(X_train=data.X_train_scaled, Y_train=data.Y_train_scaled, radius=  radius, verbose=verbose, solver=solver_name)
       
        # get training loss 
        train_loss, output = test_lin_reg(model, data.X_train_scaled, data.Y_train_scaled, criterion=criterion)
        
        # get validation loss
        val_loss, output =  test_lin_reg(model, data.X_val_scaled, data.Y_val_scaled, criterion=criterion)
        
        # get test loss
        test_loss,output = test_lin_reg(model, data.X_test_scaled, data.Y_test_scaled, criterion=criterion )

        mse = nn.MSELoss()
        train_MSE, output = test_lin_reg(model, data.X_train_scaled, data.Y_train_scaled, criterion=mse)
        
        # get validation loss
        val_MSE,output =  test_lin_reg(model, data.X_val_scaled, data.Y_val_scaled, criterion=mse)
        
        # get test loss
        test_MSE,output = test_lin_reg(model,data.X_test_scaled, data.Y_test_scaled, criterion=mse )

        end_time_model =  datetime.now() 
    
        mlflow.set_tag("model_name", "DR_linreg")
        mlflow.log_param("radius", radius)
        mlflow.log_param("bias", True)
        mlflow.log_param("benchmark_functions", func_name)
        mlflow.log_param("data", data)
        mlflow.log_param("solver", solver_name)
        mlflow.log_param("n_corrupted_points", n_corrupted_points)
        #mlflow.log_param("output_noise_prob",output_noise_prob)
        mlflow.log_param("Beta", model.Beta)
        mlflow.log_param("b", model.b)


        # Start training and testing

        mlflow.log_metric("MAE_val", val_loss)
        mlflow.log_metric("MAE_train", train_loss)
        mlflow.log_metric("MAE_test", test_loss)
        mlflow.log_metric("MSE_val", val_MSE)
        mlflow.log_metric("MSE_train", train_MSE)
        mlflow.log_metric("MSE_test", test_MSE)
        #signature = infer_signature(np.array(data.X_val_scaled, dtype=np.float64), np.array(model_torch.forward(torch.Tensor(data.X_val_scaled, dtype=torch.float64)), dtype =np.float64))
        #mlflow.pytorch.log_model(pytorch_model=model_torch, artifact_path=experiment.artifact_location)

        mlflow.log_param("training time", end_time_model - start_time_model)
        print(f'Training duration: {end_time_model - start_time_model}')

        # save model
        now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")
        str_path = f"/home/julien/Documents/code/maitrise/my version/logged_models/DR_linreg_{now_string}.csv"
        np.savetxt(
                str_path,
                np.concatenate(( model.Beta, np.array([model.b]))),
                delimiter=",",
            )
        mlflow.set_tag("path_to_model", str_path)
        
        mlflow.end_run() #end run
        
    return  {
         "status": STATUS_OK,
         "loss": val_loss
        }


def objective_FNN(params, data, criterion, experiment, n_corrupted_points, func_name, verbose):
    # unwrap hyperparameters:
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(params['batch_size'])
    learning_rate = float(params['learning_rate'])
    n_epochs = int(params['n_epochs'])
    dropout_p = float(params['dropout_p'])
    n_hidden = int(params['n_hidden'])

    # print info on run
    print("------start of trial: ------")
    print(f"batch_size = {batch_size}, learning_rate = {learning_rate}, n_hidden_neurons = {n_hidden}, n_epochs = {n_epochs}, dropout_p = {dropout_p}")
    
    # make datasets and dataloaders
    train_dataset = hm.CustomDataset(
        X=data.X_train_scaled,
        y=data.Y_train_scaled
    )

    val_dataset = hm.CustomDataset(
        X=data.X_val_scaled,
        y=data.Y_val_scaled
    )

    test_dataset = hm.CustomDataset(
        X=data.X_test_scaled,
        y=data.Y_test_scaled
    )

    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)


    # start training
    start_time_model = datetime.now() # start timer for whole training

    model = hm.FNN_Model(dropout_p=dropout_p, n_input_layers=data.X_train_scaled.shape[1], n_hidden=n_hidden, n_output_layers=1).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    with mlflow.start_run() as run:
        mlflow.set_tag("model_name", "FNN")
        mlflow.log_param("epochs", n_epochs)
        mlflow.log_param("batch size", batch_size)
        mlflow.log_param("dropout", dropout_p)
        #mlflow.log_param("random state", random_state)
        mlflow.log_param("learning rate", learning_rate)
        mlflow.set_tag("FNN", True)
        mlflow.log_param("benchmark_functions", func_name)
        mlflow.log_param("data", data)
        mlflow.log_param("n_corrupted_points", n_corrupted_points)
        mlflow.log_param("n_hidden", n_hidden)
        mlflow.log_param("solver", "Adam")
        
        # Test without training 
        print("Untrained test\n--------")
        untrained_loss = hm.test_FNN_model(val_loader, model, criterion)
        #print()
        # Start training and testing
        for ix_epoch in range(n_epochs):
            train_loss = hm.train_FNN_model(train_loader, model, criterion, optimizer=optimizer)
            val_loss = hm.test_FNN_model(val_loader, model, criterion)
            if ix_epoch % 30 == 0 and verbose:
                print(f"Epoch {ix_epoch}\n---------")
                print(f"train_loss: {train_loss}")
                print(f"test_loss: {val_loss}")
            
        mse = nn.MSELoss()
        test_mae = hm.test_FNN_model(test_loader, model, criterion)
        test_mse = hm.test_FNN_model(test_loader, model, mse)
        val_mse = hm.test_FNN_model(val_loader, model, mse)
        train_mse = hm.test_FNN_model(train_loader, model, mse)
        # Log results
        mlflow.log_metric("MAE_val", val_loss)
        mlflow.log_metric("MAE_train", train_loss)
        mlflow.log_metric("MAE_test", test_mae)
        mlflow.log_metric("MSE_val", val_mse)
        mlflow.log_metric("MSE_test", test_mse)
        mlflow.log_metric("MSE_train", train_mse)

        
        #signature_sign = infer_signature(val_dataset.X.numpy(force=True), model.forward(torch.Tensor(val_dataset.X)).numpy(force=True))
        #mlflow.pytorch.log_model(pytorch_model=model, artifact_path=experiment.artifact_location, signature = signature_sign)

        end_time_model = datetime.now()

   
        # save model
        now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")
        str_path = f"/home/julien/Documents/code/maitrise/my version/logged_models/FNN_{now_string}.pt"
        mlflow.set_tag("path_to_model", str_path)
        
        mlflow.pytorch.save_state_dict(
                model.state_dict(),
                path=str_path,
            )
        
    mlflow.end_run() #end run
    
    # return important information for hyperopt
    return  {
         "status": STATUS_OK,
         "loss": val_loss
        }

class ExhaustiveSearchError(Exception):
    pass


def validate_space_exhaustive_search(space):
    supported_stochastic_symbols = ['randint', 'quniform', 'qloguniform', 'qnormal', 'qlognormal', 'categorical']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise ExhaustiveSearchError('Exhaustive search is only possible with the following stochastic symbols: ' + ', '.join(supported_stochastic_symbols))


def suggest(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):

    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng =  np.random.default_rng(seed)#np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                })
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else (
                (key, None)) for key, value in vals.items()]))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1
            
            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id],
                                          [None], [new_result], [new_misc]))
    return rval


def eps_stability_verif(model, bound_eps:float, bound_x:float, scaler_y = None, scaler_x = None, bound_type:str="box", bigM1:float=1e3, bigM2:float=1e3, bigM3:float=1e3, slack:float = 0.0, solver:str = "SCIP", return_all:bool = False, verbose:bool = False):
    assert slack >= 0.0, "Slack must be a positive value or zero"
    assert bigM1 > 0.0, "bigM1 must be a positive value"
    assert bigM2 > 0.0, "bigM2 must be a positive value"
    assert bigM3 > 0.0, "bigM3 must be a positive value"
    assert bound_eps > 0.0, "bound_eps must be a positive value"
    assert bound_x > 0.0, "bound_x must be a positive value"
    assert bound_type in ["squared_norm", "box"], "bound_type must be either 'squared_norm' or 'box'"

    G = model.sampled_u.T
    u = np.array(model.v_opt - model.w_opt)
    Beta = np.array(u).flatten(order='F').reshape(-1,1) 
    #Beta_2 = np.vstack((Beta, Beta))

    P, d_mod = G.shape

    y = cp.Variable((1))
    y_eps = cp.Variable((1))
    x = cp.Variable((d_mod,1))
    eps = cp.Variable((d_mod, 1)) # last dimension must be constrained to zero if bias
    gamma = cp.Variable((P, 1), boolean=True)
    eta = cp.Variable((P, 1), boolean=True)
    A = cp.Variable((P, d_mod))
    B = cp.Variable((P, d_mod))
    if slack > 0:
        slack_A = cp.Variable((P, d_mod))
        slack_B = cp.Variable((P, d_mod))
    else:
        slack_A = np.zeros((P, d_mod))
        slack_B = np.zeros((P, d_mod))
    x_eps = cp.Variable((d_mod, 1))
    obj_abs = cp.Variable((1))
    delta = cp.Variable((1), boolean=True)
    #product_x = cp.Variable((P,1))
    #product_x_eps = cp.Variable((P,1))

    cstr = []

    if slack > 0:
        cstr += [slack_A <= slack]
        cstr += [slack_A >= -slack]
        cstr += [slack_B <= slack]
        cstr += [slack_B >= -slack]
    # x
    cstr += [y == Beta.T@(cp.vec(A, order='F'))] # beta = v-w donc correct avec un seul vec(A)
    #cstr += [x.T @ G.T == product_x.T]
    #for i in range(P):
    #    cstr += [product_x[i,0] <= bigM3*gamma[i,0]]
    #    cstr += [product_x[i,0] >= -bigM3*(1 - gamma[i,0])]
    cstr += [x.T @ G.T <= bigM3*gamma.T]
    cstr += [x.T @ G.T >= -bigM3*(1.0 - gamma.T)]
    for k in range(d_mod):
        for i in range(P):
            cstr += [x[k] >= A[i,k] - slack_A[i,k] + bigM1*(1-gamma[i,0])]
            cstr += [x[k] <= A[i,k] + slack_A[i,k] - bigM1*(1-gamma[i,0])]
            cstr += [A[i,k]<= bigM1*gamma[i, 0]]
            cstr += [A[i,k]>= -bigM1*gamma[i, 0]]

    # x_eps
    cstr += [y_eps == Beta.T@(cp.vec(B, order='F'))] 
    cstr += [x_eps == (x + eps)]
    cstr += [(x_eps).T @ G.T <= bigM3*eta.T] 
    cstr += [(x_eps).T @ G.T >= -bigM3*(1.0 - eta.T)]   
    #cstr += [x_eps.T @ G.T == product_x_eps.T]
    #for i in range(P):
    #    cstr += [product_x_eps[i,0] <= bigM3*eta[i,0]]
    #    cstr += [product_x_eps[i,0] >= -bigM3*(1 - eta[i,0])]
    for k in range(d_mod):
        for i in range(P):
            cstr += [x_eps[k] >= B[i,k] - slack_B[i,k] + bigM1*(1-eta[i, 0])]
            cstr += [x_eps[k] <= B[i,k] + slack_B[i,k] - bigM1*(1-eta[i, 0])]
            cstr += [B[i,k]<= bigM1*eta[i, 0]]
            cstr += [B[i,k]>= -bigM1*eta[i, 0]]

    # must consider model bias
    if model.bias:
        cstr += [eps[-1,:] == 0.0]
        cstr += [x[-1,:]== 1.0]

    # bounds
    if bound_type == "squared_norm":
        cstr += [cp.sum_squares(eps) <= bound_eps**2]
        cstr += [cp.sum_squares(x) <= bound_x**2]
        cstr += [cp.sum_squares(x_eps) <= bound_x**2]
    elif bound_type == "box" or bound_type != "squared_norm":
        cstr += [eps <= bound_eps]
        cstr += [eps >= -bound_eps]
        cstr += [x <= bound_x]
        cstr += [x >= -bound_x]


    # loop hole for absolute value objective
    cstr += [obj_abs >= (y_eps - y)]
    cstr += [obj_abs >= -(y_eps - y)]
    cstr += [y_eps - y <= bigM2*delta]
    cstr += [y_eps - y >= -bigM2*(1-delta)]
    cstr += [obj_abs <= y_eps - y + bigM2*(1-delta)]
    cstr += [obj_abs <= -(y_eps - y) + bigM2*delta]
    cstr += [obj_abs >= 0.0]
    cstr += [obj_abs <= 1.0e10]


    obj = obj_abs

    prob = cp.Problem(cp.Maximize(obj), cstr)
    prob.solve(solver=solver, verbose =verbose)

    assert prob.status == cp.OPTIMAL, "Optimization problem did not converge"
    if verbose:
        print(x.value.T @ G.T)
        print(gamma.value.T)
        print(x_eps.value.T @ G.T)
        print(eta.value.T)
        print(y.value)
        print(y_eps.value)
    # return the scaled value of the objective function
    if scaler_y is not None and scaler_x is not None:
        y = scaler_y.inverse_transform(np.reshape(y.value, (1,1)))
        y_eps = scaler_y.inverse_transform(np.reshape(y_eps.value, (1,1)))
        eps = np.abs(scaler_x.inverse_transform(x_eps.value[0:-1,:].T) - scaler_x.inverse_transform(x.value[0:-1,:].T))
        
        return np.abs(y - y_eps), y, y_eps, eps if return_all else np.abs(y - y_eps)
    else:
        return np.abs(y.value - y_eps.value), y.value, y_eps.value, eps.value if return_all else np.abs(y.value - y_eps.value)
        

class fake_scaler:
    def __init__(self):
        pass
    def inverse_transform(self, x):
        return x
    def transform(self, x):
        return x
    def fit_transform(self, x):
        return x
    
def stability_objective_scnn(params, data, solver_name, experiment, wasserstein, dataset_name, verbose, bound_eps, bound_type, bigM1, bigM2, bigM3):
    # unwrap hyperparameters:
    radius = float(params['radius'])
    max_neurons = int(params['max_neurons'])
    #wasserstein = str(params['wasserstein'])
    #bound_type = str(params['bound_type']) # 'box' or 'squared_norm'
    bias = True
    

    # print info on run
    print("------start of trial: ------")
    print(f"radius= {radius}, max_neurons = {max_neurons}, bias = {bias}")


    start_time_model = datetime.now() # start timer for whole training
    # define model
    model = hm.wadiro_scnn()
    max_norm = np.linalg.norm(data['X_train_scaled'], axis=1).max()
    
    with mlflow.start_run() as run:
       
       
        # train
        model.train(X_train=data['X_train_scaled'], Y_train=data['Y_train_scaled'], radius = radius, bias = bias, max_neurons=max_neurons, verbose=verbose, solver=solver_name, wasserstein=wasserstein,)
        end_time_model =  datetime.now() 
        # torch model
        y_pred_train = model.predict_with_sampled_u(data['X_train_scaled'])
        y_pred_test = model.predict_with_sampled_u(data['X_test_scaled'])

        mean_absolute_error_train = sk.metrics.mean_absolute_error(data['Y_train'], data['scaler_y'].inverse_transform(y_pred_train))
        mean_absolute_error_test = sk.metrics.mean_absolute_error(data['Y_test'], data['scaler_y'].inverse_transform(y_pred_test))

        root_mean_squared_error_train = sk.metrics.root_mean_squared_error(data['Y_train'], data['scaler_y'].inverse_transform(y_pred_train))
        root_mean_squared_error_test = sk.metrics.root_mean_squared_error(data['Y_test'], data['scaler_y'].inverse_transform(y_pred_test))
        
        # verification of stability
        #bound_type = "squared_norm" # 'box' or 'squared_norm'
        bound_x = 5 if bound_type == "box" else 2*max_norm

        try:
            stability, y, y_eps, eps = eps_stability_verif(model, bound_eps=bound_eps, bound_x=bound_x, scaler_y=data['scaler_y'], scaler_x=data['scaler_x'], bound_type=bound_type, bigM1=bigM1,bigM2=bigM2, bigM3=bigM3, slack=0, solver=solver_name, return_all=True, verbose=verbose)
        except:
            stability = np.nan
            eps = np.nan

        mlflow.log_param('eps_norm', np.linalg.norm(eps))
        mlflow.log_param('eps', eps)
        mlflow.set_tag("model_name", f"wadiro_scnn_{wasserstein}")
        mlflow.log_param("radius", radius)
        mlflow.log_param("max_neurons", max_neurons)
        mlflow.log_param("bias", bias)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("data", data)
        mlflow.log_param("solver", solver_name)
        mlflow.log_param("wasserstein", wasserstein)
        mlflow.log_param("bound_eps", bound_eps)
        mlflow.log_param("bound_type", bound_type)
        mlflow.log_param("bigM1", bigM1)
        mlflow.log_param("bigM2", bigM2)
        mlflow.log_param("bigM3", bigM3)
        mlflow.log_param("max_norm", max_norm)
        mlflow.log_param("bound_type", bound_type)
        mlflow.log_param("bound_x", bound_x)

        # Start training and testing
        mlflow.log_metric('stability', stability)
        mlflow.log_metric("MAE_train", mean_absolute_error_train)
        mlflow.log_metric("MAE_test", mean_absolute_error_test)
        mlflow.log_metric("RMSE_train", root_mean_squared_error_train)
        mlflow.log_metric("RMSE_test", root_mean_squared_error_test)
        #signature = infer_signature(np.array(data.X_val_scaled, dtype=np.float64), np.array(model_torch.forward(torch.Tensor(data.X_val_scaled, dtype=torch.float64)), dtype =np.float64))
        #mlflow.pytorch.log_model(pytorch_model=model_torch, artifact_path=experiment.artifact_location)

        mlflow.log_param("training time", end_time_model - start_time_model)
        print(f'Training duration: {end_time_model - start_time_model}')

        # save model
        now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")       
        mlflow.end_run() #end run
        
    return  {
         "status": STATUS_OK,
         "loss": mean_absolute_error_test
        }