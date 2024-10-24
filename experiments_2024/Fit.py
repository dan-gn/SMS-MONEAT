import numpy as np
from numpy.random import rand
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import seed
from numpy.random import randint
from sklearn.model_selection import cross_val_score
import math


def fit(xtrain, ytrain, kk):
    #  print('x_train', xtrain)
    #  print('y_train', ytrain)
    #  print('kk', kk)
     k = 5
     if len(kk) == 0:
         seed(1)
         kk= randint(1, np.size(xtrain, 1))
    #print(sf)
     sf= []
     pos=[]
     groups=[]
     for i in range(0, np.size(xtrain, 1)):
         if(kk[i]==1):
             sf.append(i) 

    
     pos=np.transpose(sf)
    #  print('pos', pos)
     kf = KFold(n_splits=k)

     model  = KNeighborsClassifier(n_neighbors =1,metric='euclidean') 
     X=xtrain[:,pos]
     
    #  model.fit(X, ytrain)
    #  score = model.predict(X)
    #  print('y_predict', score)

    #  print('X', X)
     scores = cross_val_score(model, X, ytrain, cv=5)

    #  print('scores', scores) 
     cost = sum(scores)/k

    #  print('cost', cost) 
     return cost*100

