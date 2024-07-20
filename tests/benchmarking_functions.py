
import sys
sys.path.append('../wadiro_ml')
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