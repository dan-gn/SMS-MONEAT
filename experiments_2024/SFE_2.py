import numpy as np
import math

import pandas as pd
import scipy.stats as stats
from Fit import fit

class SFE:

    def __init__(self, problem: dict, params: dict) -> None:
        # Execution algorithm parameters
        self.max_iterations = params['max_iterations']
        self.UR = 0.3
        self.UR_Max = 0.3
        self.UR_Min = 0.001
        
        self.Run = 1



        # Problem parameters
        self.x_train, self.y_train = problem['x_train'], problem['y_train']
        # self.X_test, self.y_test = problem['x_test'], problem['y_test']
        

    def run(self, seed: int = None, debug: bool = False):
        np.random.seed(seed)

        EFs = 1
    
        individual = np.random.randint(0, 2, np.size(self.x_train, 1))   # Initialize an Individual X
        Fit_X = fit(self.x_train, self.y_train, individual)                    # Calculate the Fitness of X
        Nvar = np.size(self.x_train, 1)                         # Number of Features in Dataset

    
        while (EFs <= self.max_iterations):
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
                SN = 1                                      # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X, SN)        # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                new_individual = np.copy(individual)
                new_individual[K] = 1                                # Set X_New (K)=1

            # print(Input, Target, X_New)
            Fit_X_New = fit(self.x_train, self.y_train, new_individual)             # Calculate the Fitness of X_New
            # print(Fit_X_New)
            if Fit_X_New > Fit_X:
                X = np.copy(new_individual)
                Fit_X = Fit_X_New

            UR = (self.UR_Max-self.UR_Min)*((self.max_iterations-EFs)/self.max_iterations)+self.UR_Min  # Eq(3)
            print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format( EFs, Fit_X, np.sum(new_individual), self.Run))
            EFs = EFs+1

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