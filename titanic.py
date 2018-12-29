# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:20:45 2018

@author: chaitu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as md
dataset1 =pd.read_csv('train.csv')
dataset2 =pd.read_csv('test.csv')
dataset2.isnull().sum()
size_mapping={'C':1,'Q':2,'S':3}
r={'C':1,'Q':2,'S':3}
dataset1['Embarked']=dataset1['Embarked'].map(size_mapping)
dataset2['Embarked']=dataset2['Embarked'].map(r)
xtest=dataset2.iloc[:,:].values
xtrain =dataset1.iloc[:,:-1].values
ytrain=dataset1.iloc[:,-1].values


from sklearn.preprocessing import Imputer
imp =Imputer(strategy='most_frequent')
imp1 =Imputer(strategy='mean')
xtrain[:,3]=(imp1.fit_transform(xtrain[:,3].reshape(-1,1))).reshape(1,-1)
xtrain[:,7]=(imp.fit_transform(xtrain[:,7].reshape(-1,1))).reshape(1,-1)
xtest[:,[2,5]]=imp1.fit_transform(xtest[:,[2,5]])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
l = LabelEncoder()
l=l.fit(xtrain[:,2])
xtrain[:,2] =l.transform(xtrain[:,2])
l=l.fit(xtest[:,1])
xtest[:,1] =l.transform(xtest[:,1])
one1=OneHotEncoder(categorical_features=[1,6])
one=OneHotEncoder(categorical_features=[2,7])
xtest=one1.fit_transform(xtest).toarray()
xtrain = one.fit_transform(xtrain).toarray()
xtrain=np.delete(xtrain,np.s_[0,2],axis=1)
xtrain=np.delete(xtrain,np.s_[3],axis=1)
xtest=np.delete(xtest,np.s_[0,2],axis=1)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=2)
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, leaf_size=1,metric = 'minkowski', p = 2)
classifier.fit(xtrain, ytrain)
ypredknn=classifier.predict(xtest)



from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',C=0.08,gamma=0.02 ,random_state = 0)
classifier.fit(xtrain, ytrain)

# Predicting the Test set results
ypredksvm = classifier.predict(xtest)

from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=14, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
classifier1.fit(xtrain,ytrain)

# Predicting the Test set results
y_pred = classifier1.predict(xtest)



n_neighbors = [18,10]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,26))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=md.GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")
gd.fit(xtrain, ytrain)
print(gd.best_score_)
print(gd.best_estimator_)

C=list(np.arange(0.1,1,0.1))
gamma=list(np.arange(0.01,0.025,0.01))
n=list(range(1,20,1))
hyperparms={'n_estimators':n}
gd=md.GridSearchCV(estimator=RandomForestClassifier(),param_grid=hyperparms)
gd.fit(xtrain,ytrain)
print(gd.best_score_)
print(gd.best_estimator_)

