import sys
sys.path.append('.')
import hyperopt as hp
import numpy as np
import scipy as sc
import comparison_models as hm
import benchmark_functions as bf
import benchmarking_functions as bench
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import mlflow
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from datetime import datetime
import torch.nn as nn
import cvxpy as cp
import mlflow.pytorch
from mlflow.models.signature import infer_signature, set_signature
import hyperopt
from functools import partial
import pandas as pd
import seaborn as sns
from mlflow.exceptions import MlflowException
MAX_NEURONS = 300
MIN_NEURONS = 5



search_space_DR_SCNN = {'radius': hp.loguniform('radius',-8, 3),
                'max_neurons': hp.quniform('max_neurons', MIN_NEURONS, MAX_NEURONS, 1),
                'wasserstein': hp.choice('wasserstein', ["l1", "l2"]),
                }

search_space_SCNN = {'lamb_reg': hp.loguniform('lamb_reg',-8, 3),
                'max_neurons': hp.quniform('max_neurons', MIN_NEURONS, MAX_NEURONS, 1),
                'regularizer': hp.choice('regularizer', ["LASSO", "RIDGE"]),
                }

search_space_SCNN_no_reg = {#'lamb_reg': hp.loguniform('lamb_reg',-8, 1),
                'max_neurons': hp.quniform('max_neurons', MIN_NEURONS, MAX_NEURONS, 1),
                }


search_space_DR_linreg = {'radius': hp.loguniform('radius',-8, 3),
                          'wasserstein': hp.choice('wasserstein', ["l1", "l2"]),
                }

search_space_linreg = {'lamb_reg': hp.loguniform('lamb_reg',-8, 3),
                'regularizer': hp.choice('regularizer', ["LASSO","RIDGE"]),
                }

search_space_FNN =  {'batch_size' : hp.quniform('batch_size', 2, 100, 1),
                'learning_rate': hp.loguniform('learning_rate',-8, 1),
                'n_hidden' : hp.quniform('n_hidden',MIN_NEURONS, MAX_NEURONS, 1),
                'n_epochs': hp.quniform( 'n_epochs',1, 1000, 1),
                'dropout_p': hp.uniform('dropout_p', 0.01, 0.4)}


rng = np.random.default_rng(42)
criterion = nn.L1Loss()

solver_name = "CLARABEL"
n_runs = 10
max_evals = 400
max_evals_FNN = int(max_evals)
verbose = False
N = 2000
train_size = 0.75
test_size = 0.2
corruption_percentage_list = [ 0.15, 0.30, 0.45, 0.60]
n_dim = 4

FUNCTIONS_LIST = [ 
	[bf.Ackley, n_dim, "Ackley"],			
	[bf.Keane,n_dim, "Keane"],
	[bf.PichenyGoldsteinAndPrice,2, "PichenyGoldsteinAndPrice"],
	[bf.McCormick, 2, "McCormick"], # 2 dims
        [bf.Himmelblau, 2, "Himmelblau"],# 2 dims
        [bf.Rosenbrock, n_dim, "Rosenbrock"],
	]

for corruption_percentage in corruption_percentage_list:
        n_corrupted_points = int(N*(1-test_size)*corruption_percentage) 
        for run in range(n_runs):
                try:
                        experiment = mlflow.get_experiment_by_name(f"large_test_CC_run{run}_corper{str(corruption_percentage)[2:]}")
                        experiment_id = experiment.experiment_id
                except AttributeError:
                        experiment_id = mlflow.create_experiment(f"large_test_CC_run{run}_corper{str(corruption_percentage)[2:]}", artifact_location='./mlruns/')
                        experiment = mlflow.get_experiment_by_name(f"large_test_CC_run{run}_corper{str(corruption_percentage)[2:]}")

                #experiment = mlflow.set_experiment(f"large_test_CC_run{run}_corper{corruption_percentage}")
                # Get Experiment Details
                print("Experiment_id: {}".format(experiment.experiment_id))
                print("Artifact Location: {}".format(experiment.artifact_location))
                print("Name: {}".format(experiment.name))
                print("Tags: {}".format(experiment.tags))
                print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

                for benchmark_func in FUNCTIONS_LIST:
                        data = bench.dataset_wrapper(benchmark_func=benchmark_func[0], n_dim=benchmark_func[1], rng =rng, N = N, train_size=train_size, test_size=test_size)
                        func_name = benchmark_func[2]
                        data.generate_data_wasserstein_corrupt_both(corrupt_data_points= n_corrupted_points, min_distance=0.05, n_projections=100, k_multiplier=1.5, L=2, seed=42)


                        print(f"SCNN_no_reg on {func_name}: \n")
                        fmin_SCNN_no_reg = partial(bench.objective_SCNN_no_reg, data = data, criterion = criterion, solver_name = solver_name, experiment=experiment, n_corrupted_points = n_corrupted_points, func_name=func_name, verbose=verbose)
                        argmin_SCNN_no_reg = fmin(fn=fmin_SCNN_no_reg,
                                space=search_space_SCNN_no_reg,
                                algo= hyperopt.anneal.suggest, #hyperopt.tpe.suggest, # try anneal.suggest
                                max_evals=int(max_evals)) #trials=spark_trials)
                        
                        print(f"WaDiRO SCNN on {func_name}: \n")
                        fmin_DR_SCNN = partial(bench.objective_DR_SCNN, data = data, criterion = criterion, solver_name = solver_name, experiment=experiment, n_corrupted_points=n_corrupted_points, func_name=func_name, verbose=verbose)
                        argmin_DR_SCNN = fmin(fn=fmin_DR_SCNN,
                                space=search_space_DR_SCNN,
                                algo= hyperopt.anneal.suggest, #hyperopt.tpe.suggest, # try anneal.suggest
                                max_evals=max_evals) #trials=spark_trials)
                        
                        print(f"SCNN on {func_name}: \n")
                        fmin_SCNN = partial(bench.objective_SCNN, data = data, criterion = criterion, solver_name = solver_name, experiment=experiment, n_corrupted_points = n_corrupted_points, func_name=func_name, verbose=verbose)
                        argmin_SCNN = fmin(fn=fmin_SCNN,
                                space=search_space_SCNN,
                                algo= hyperopt.anneal.suggest, #hyperopt.tpe.suggest, # try anneal.suggest
                                max_evals=max_evals) #trials=spark_trials)
                        
                        print(f"linreg on {func_name}: \n")
                        fmin_linreg = partial(bench.objective_linreg, data = data, criterion = criterion, solver_name = solver_name, experiment=experiment, n_corrupted_points=n_corrupted_points,  func_name=func_name, verbose=verbose)
                        argmin_linreg = fmin(fn=fmin_linreg,
                                space=search_space_linreg,
                                algo= hyperopt.anneal.suggest,#hyperopt.tpe.suggest, # try anneal.suggest
                                max_evals=max_evals) #trials=spark_trials)

                        print(f"WaDiRo linreg on {func_name}: \n")
                        fmin_DR_linreg = partial(bench.objective_DR_linreg, data = data, criterion = criterion, solver_name = solver_name, experiment=experiment, n_corrupted_points=n_corrupted_points,  func_name=func_name, verbose=verbose)
                        argmin_DR_linreg = fmin(fn=fmin_DR_linreg,
                                space=search_space_DR_linreg,
                                algo= hyperopt.anneal.suggest,#hyperopt.tpe.suggest, # try anneal.suggest
                                max_evals=max_evals) #trials=spark_trials)
                                
                        
                        print(f"FNN on {func_name}: \n")
                        fmin_FNN = partial(bench.objective_FNN, data = data, criterion = criterion, experiment=experiment, n_corrupted_points=n_corrupted_points,  func_name=func_name, verbose=verbose)
                        argmin_FNN = fmin(fn=fmin_FNN,
                                space=search_space_FNN,
                                algo= hyperopt.anneal.suggest, # try anneal.suggest
                                max_evals=max_evals_FNN) #trials=spark_trials)