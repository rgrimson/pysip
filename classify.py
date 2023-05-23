#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt


#%%
################################
## Claificadores Supervisados ##
################################

###################
## Random Forest ##
###################
# Empiezo con RF
from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import cohen_kappa_score as kappa
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
#Lo sistematizo en funciones
# Búsqueda de parámetros max_depth y n_estimators

#%%
def find_best_RF_max_depth(X, Y, init_md = 20, init_ne = 100, n_jobs = 15, n_splits = 5):
    lmd = np.arange(init_md//2,init_md*2)
    print( 'Buscando el valor óptimo para el parámetro max_depth:')
    print(f'    Rango: {init_md//2} a {init_md*2}.')
    print(f'    init_md: {init_md}, init_ne: {init_ne}, n_splits: {n_splits}.')
    print(f'    n_datos: {X.shape[0]}, n_features: {X.shape[1]}.\n')
          
    
    accTest = np.zeros([len(lmd),n_splits])
    accTrain = np.zeros([len(lmd),n_splits])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):
        print(f'fold {j+1} de {n_splits}.',end='\n')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        for i,md in tqdm(enumerate(lmd)):
            clf = RandomForestClassifier(n_estimators = init_ne, max_depth = md,n_jobs=n_jobs)
            #clf = tree.DecisionTreeClassifier(max_depth = md)
            clf = clf.fit(X_train, y_train)
            accTest[i,j] = accuracy_score(y_test, clf.predict(X_test))
            accTrain[i,j] = accuracy_score(y_train, clf.predict(X_train))
    #%
    print()
    #Calculo promedios de los K folds
    scoresTest = accTest.mean(axis=1)
    scoresTrain = accTrain.mean(axis=1)
    
    #Busco el mejor resultado
    imx = np.argmax(scoresTest)
    best_prof = lmd[imx]
    best_acc = scoresTest[imx]
    print(f'El mejor resultado en el conjunto de testeo fue obtenido para un arbol de profundidad {best_prof} y una exactitud de {best_acc:.3f}')
    imx = np.where(scoresTest>best_acc-0.001)[0][0]
    best_prof = lmd[imx]
    best_acc = scoresTest[imx]
    print(f'Me quedo con una profundidad de {best_prof} y una exactitud de {best_acc:.3f} (ne={init_ne}).')
    
    #%
    #Grafico promedios
    plt.figure()
    plt.plot(lmd,scoresTest,label="Test")
    #print(f'El mejor resultado fue con un árbol de profundidad {lmd[np.argmax(scoresTest)]}: {scoresTest[np.argmax(scoresTest)]:.3f}')
    plt.plot(lmd,scoresTrain,label="Train")
    plt.title(f"Exactitud en función de la profundidad (ne={init_ne})")    
    plt.ylabel("Exactitud mediana")
    plt.xlabel("Profundidad del árbol de decisión")
    plt.axvline(x=best_prof, label='Mejor profundidad', c='r')
    plt.legend()
    #plt.ylim(0.99,1.0001)
    plt.show()
    return best_prof

                    
#%%
#N-ESTIMATORS
def find_best_RF_n_estimators(X, Y, init_md = 20, init_ne = 100, n_jobs = 15, n_splits = 5):
    print( 'Buscando el valor óptimo para el parámetro n_estimators:')
    print(f'    Rango: {init_ne//2} a {2*init_ne}.')
    print(f'    init_md: {init_md}, init_ne: {init_ne}, n_splits: {n_splits}.')
    print(f'    n_datos: {X.shape[0]}, n_features: {X.shape[1]}.\n')

    lne = np.arange(init_ne//2,init_ne*2)        
    accTest_ne = np.zeros([len(lne),n_splits])
    accTrain_ne = np.zeros([len(lne),n_splits])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):
        print(f'fold {j+1} de {n_splits}.',end='\n')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        for i, ne in tqdm(enumerate(lne)):
            clf = RandomForestClassifier(max_depth = init_md, n_jobs=n_jobs, n_estimators=ne)
            #clf = tree.DecisionTreeClassifier(max_depth = md)
            clf = clf.fit(X_train, y_train)
            accTest_ne[i,j] = accuracy_score(y_test, clf.predict(X_test))
            accTrain_ne[i,j] = accuracy_score(y_train, clf.predict(X_train))
    #%
    print()
    #Calculo promedios de los K folds
    scoresTest = accTest_ne.mean(axis=1)
    scoresTrain = accTrain_ne.mean(axis=1)
    
    #Busco el mejor resultado
    imx = np.argmax(scoresTest)
    best_ne = lne[imx]
    best_acc = scoresTest[imx]
    print(f'El mejor resultado en el conjunto de testeo fue obtenido para un n_estimators {best_ne} y una exactitud de {best_acc:.3f}')
    imx = np.where(scoresTest>best_acc-0.001)[0][0]
    best_ne = lne[imx]
    best_acc = scoresTest[imx]
    print(f'Me quedo con un n_estimators {best_ne} y una exactitud de {best_acc:.3f} (md={init_md}).')
    #%
    #Grafico promedios
    plt.figure()
    plt.plot(lne,scoresTest,label="Test")
    #print(f'El mejor resultado fue con {lne[np.argmax(scoresTest)]}: {scoresTest[np.argmax(scoresTest)]:.3f}')
    plt.plot(lne,scoresTrain,label="Train")
    plt.title(f"Exactitud en función de la cantidad de arboles (md={init_md})")
    plt.ylabel("Exactitud mediana")
    plt.xlabel("Cantidad de árboles de decisión")
    plt.axvline(x=best_ne, label='Mejor profundidad', c='r')
    plt.legend()
    #plt.ylim(0.99,1.0001)
    plt.show()
    return best_ne
    
