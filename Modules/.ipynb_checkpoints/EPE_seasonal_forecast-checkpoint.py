import numpy as np
# from netCDF4 import Dataset
# import glob
# import matplotlib.pyplot as plt
import scipy.stats as sp
import pandas as pd
from sklearn import preprocessing
# from sklearn.metrics import mean_squared_error
# from math import cos, asin, sqrt, pi
# import os

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(1, '../Modules/')
from EPE_board_visualisation import *


### Produce forecasts based on optimal predictors ###

def forecast(target_indicator, first_train, last_train, sol, mod, pred_dataframe):

    # define the solution (predictors combination) to use for the forecast
    array_best=sol

    # define the target variable
    target_dataset=target_indicator

    # define the sequence_length, final_sequence, and feature_selection from the solution array
    split=int(array_best.shape[0]/3)
    
    sequence_length_best = array_best[0:split]
    final_sequence_best = array_best[split:split*2]
    feat_sel_best = array_best[split*2:]

    # define the number of columns and rows for the board
    n_cols = split
    n_rows = int((sequence_length_best + final_sequence_best).max())+10
    
    board_best = create_board(n_rows, n_cols, final_sequence_best, sequence_length_best, feat_sel_best)

    # create a solution in which all variables are selected with no lag and maximum sequence length
    time_lags = np.repeat(0,pred_dataframe.shape[1])
    time_sequences = np.repeat(n_rows,n_cols)
    variable_selection = np.repeat(1,pred_dataframe.shape[1])
    
    # create a df with target values and all predictors including all lags
    # (note, here target dataset dates are crucial)
    dataset_opt = target_dataset.copy()
    for i,col in enumerate(pred_dataframe.columns):
        if variable_selection[i] == 0:
            continue
        for j in range(time_sequences[i]):
            dataset_opt[str(col)+'_lag'+str(time_lags[i]+j)] = pred_dataframe[col].shift(time_lags[i]+j)

    # split in train and test set   
    first_train_index=int(np.argwhere(target_indicator.index==first_train))
    last_train_index=int(np.argwhere(target_indicator.index==last_train))
    
    train_dataset_opt = dataset_opt[first_train_index:last_train_index]
    test_dataset_opt = dataset_opt[last_train_index:]

    Y_column = 'Target' 
           
    X_train=train_dataset_opt[train_dataset_opt.columns.drop([Y_column]) ]
    Y_train=train_dataset_opt[Y_column]
    X_test=test_dataset_opt[test_dataset_opt.columns.drop([Y_column]) ]
    Y_test=test_dataset_opt[Y_column]

    # standardize data
    scaler = preprocessing.StandardScaler()
    X_std_train = scaler.fit(X_train)

    X_std_train = scaler.transform(X_train)
    X_std_test = scaler.transform(X_test)

    X_train=pd.DataFrame(X_std_train,columns=X_train.columns,index=X_train.index)
    X_test=pd.DataFrame(X_std_test,columns=X_test.columns,index=X_test.index)
    
    # select features based on the input board
    X_train_new = np.array(X_train)[:,board_best.T.reshape(1,-1)[0].astype(bool)]
    X_test_new = np.array(X_test)[:,board_best.T.reshape(1,-1)[0].astype(bool)]

    # train
    clf = mod
    clf.fit(X_train_new, Y_train)
    # test
    predictions = clf.predict(X_test_new)
    train_predictions = clf.predict(X_train_new)
    Y_pred = pd.DataFrame(predictions, columns=['Y_pred'], index=Y_test.index)
    
    return clf.__class__.__name__,predictions, train_predictions # full model name, 1D output forecast (year), 1D output for train data (year)
