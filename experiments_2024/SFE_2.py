import numpy as np
import math

import pandas as pd
import scipy.stats as stats
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from utilities.stats_utils import geometric_mean
from utilities.fitness_functions import torch_fitness_function

class SFE:

    def __init__(self, problem: dict, params: dict) -> None:
        # Execution algorithm parameters
        self.max_iterations = params['max_iterations']
        self.UR = params['UR']
        self.UR_Max = params['UR_max']
        self.UR_Min = params['UR_min']
        self.SN = params['SN']
        
        self.Run = 1



        # Problem parameters
        self.x_train, self.y_train = problem['x_train'], problem['y_train']
        self.x_test, self.y_test = problem['x_test'], problem['y_test']
        

    def run(self, seed: int = None, debug: bool = False):
        np.random.seed(seed)
        Nvar = np.size(self.x_train, 1)                         # Number of Features in Dataset

        individual = np.random.randint(0, 2, np.size(self.x_train, 1))   # Initialize an Individual X
        # Fit_X = fit(self.x_train, self.y_train.squeeze(1), individual)                    # Calculate the Fitness of X
        _, fitness, _ = self.evaluate(individual, self.x_train, self.y_train)
        Fit_X = fitness[0]
        EFs = 1
    
        while (EFs < self.max_iterations):
            new_individual = np.copy(individual)
            # Non-selection operation:

            U_Index = np.where(individual == 1)                      # Find Selected Features in X
            NUSF_X = np.size(U_Index, 1)                    # Number of Selected Features in X
            UN = math.ceil(self.UR*Nvar)                         # The Number of Features to Unselect: Eq(2)
            # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
            # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
            K1 = np.random.randint(0, NUSF_X, UN)           # Generate UN random number between 1 to the number of slected features in X
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]                          # K=index(U)
            new_individual[K] = 0                                    # Set X_New (K)=0 


            # Selection operation:
            if np.sum(new_individual) == 0:
                S_Index = np.where(individual == 0)                  # Find non-selected Features in X
                NSF_X = np.size(S_Index, 1)                 # Number of non-selected Features in X
                self.SN = 1                                      # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X, self.SN)        # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                new_individual = np.copy(individual)
                new_individual[K] = 1                                # Set X_New (K)=1

            # print(Input, Target, X_New)
            # Fit_X_New = fit(self.x_train, self.y_train.squeeze(1), new_individual)             # Calculate the Fitness of X_New
            _, new_fitness, _ = self.evaluate(new_individual, self.x_train, self.y_train)
            Fit_X_New = new_fitness[0]
            # print(Fit_X_New)
            if Fit_X_New <= Fit_X:
                individual = np.copy(new_individual)
                Fit_X = Fit_X_New

            UR = (self.UR_Max-self.UR_Min)*((self.max_iterations-EFs)/self.max_iterations)+self.UR_Min  # Eq(3)
            # print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format( EFs, Fit_X, np.sum(new_individual), self.Run))
            EFs = EFs+1
        self.individual = np.copy(individual)

    def evaluate(self, member, x, y, n_folds = 3):
        features_selected = [i for i, xi in enumerate(member) if xi == 1]
        features_selected = torch.tensor(features_selected)
        if features_selected.shape[0] < 1:
            return None, np.array([math.inf, math.inf]), 0
        x_prima = x.index_select(1, features_selected)
        min_class = min(int(torch.sum(y)), y.shape[0] - int(torch.sum(y)))
        k = min(min_class, n_folds)
        loss = np.zeros(k)
        acc = np.zeros(k)
        g_mean = np.zeros(k)
        skf = StratifiedKFold(n_splits=k)
        for i, (train_index, test_index) in enumerate(skf.split(x_prima, y)):
            model = KNeighborsClassifier(n_neighbors=2)
            model.fit(x_prima[train_index], y[train_index].ravel())
            y_predict = torch.tensor(model.predict(x_prima[test_index]))
            y_real = y[test_index].squeeze(dim=1)
            loss[i] = torch_fitness_function(y_real, y_predict) 
            acc[i] = (y_real == torch.round(y_predict)).type(torch.float32).mean()
            g_mean[i] = geometric_mean(y_real, y_predict)
        return acc.mean(), [loss.mean(), features_selected.shape[0]], g_mean.mean()

   


    def final_evaluate(self, member, x_train, y_train, x_test, y_test):
        features_selected = [i for i, xi in enumerate(member) if xi == 1]
        features_selected = torch.tensor(features_selected)
        if features_selected.shape[0] < 1:
            return None, np.array([math.inf, math.inf]), 0
        x_train_prima = x_train.index_select(1, features_selected)
        x_test_prima = x_test.index_select(1, features_selected)
        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(x_train_prima, y_train.ravel())
        y_predict = torch.tensor(model.predict(x_test_prima))
        y_real = y_test.squeeze(dim=1)
        loss = torch_fitness_function(y_real, y_predict) 
        acc = (y_real == torch.round(y_predict)).type(torch.float32).mean()
        g_mean = geometric_mean(y_real, y_predict)
        return acc, [loss, features_selected.shape[0]], g_mean





if __name__ == '__main__':

    data = pd.read_csv("SFE Python Code\\colon.csv")
    data = data.values
    Input = np.asarray(data[:, 0:-1])
    Input = stats.zscore(Input)
    Target = np.asarray(data[:, -1])




    problem = {
        'x_train' : Input,
        'y_train' : Target
    }
    params = {'max_iterations': 60}
    alg = SFE(problem, params)
    alg.run(seed = 0)