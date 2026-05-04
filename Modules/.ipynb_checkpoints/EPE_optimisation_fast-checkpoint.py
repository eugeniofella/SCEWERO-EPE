import numpy as np
import pandas as pd

from PyCROSL.AbsObjectiveFunc import AbsObjectiveFunc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def solution_to_selected_cols(solution, p, col_index, max_shift):
    time_sequences = np.array(solution[:p]).astype(int)
    time_lags      = np.array(solution[p:2*p]).astype(int)
    variable_sel   = np.array(solution[2*p:3*p]).astype(int)

    selected_cols = []
    for i in range(p):
        if variable_sel[i] == 0:
            continue
        win = int(time_sequences[i])
        if win <= 0:
            continue
        start = int(time_lags[i])
        for j in range(win):
            lag = start + j
            if 1 <= lag <= max_shift:
                selected_cols.append(col_index[(i, lag)])
    return selected_cols

class optimisation_fast(AbsObjectiveFunc):
    """
    This is the constructor of the class, here is where the objective function can be setted up.
    In this case we will only add the size of the vector as a parameter.
    """
    def __init__(self, size, pred_dataframe, MAX_LAG, MAX_WINDOW,
                fitness_cache, col_index, X_train_full, y_train_full, X_test_full, y_test_full, mu, 
                 sigma, indiv_file):
        
        self.size = size
        self.opt = "min" # it can be "max" or "min"

        self.pred_dataframe = pred_dataframe

        self.MAX_LAG = MAX_LAG
        self.MAX_WINDOW = MAX_WINDOW
        self.MAX_SHIFT = MAX_LAG + MAX_WINDOW
        self.fitness_cache = fitness_cache
        self.col_index = col_index
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full
        self.X_test_full = X_test_full
        self.y_test_full = y_test_full
        self.mu = mu
        self.sigma = sigma
        self.indiv_file = indiv_file

        # We set the limits of the vector (window size, time lags and variable selection)
        # array where each component indicates the maximum value of the component of the vector
        self.sup_lim = np.append(np.append(np.repeat(self.MAX_WINDOW, self.pred_dataframe.shape[1]),np.repeat(self.MAX_LAG, self.pred_dataframe.shape[1])),np.repeat(1, self.pred_dataframe.shape[1]))
        # array where each component indicates the minimum value of the component of the vector
        self.inf_lim = np.append(np.append(np.repeat(1, self.pred_dataframe.shape[1]),np.repeat(0, self.pred_dataframe.shape[1])),np.repeat(0, self.pred_dataframe.shape[1])) 
        
        # we call the constructor of the superclass with the size of the vector
        # and wether we want to maximize or minimize the function 
        super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)
    
    """
    This will be the objective function, that will recieve a vector and output a number
    """
    def objective(self, solution):
        # Read data
        sol_file = pd.read_csv(self.indiv_file,sep=' ',header=0)
 
        key = tuple(solution)
        if key in self.fitness_cache: # If the solution has been computed before, return the cached value
            return self.fitness_cache[key]



        selected_cols = solution_to_selected_cols(
            solution, 
            self.pred_dataframe.shape[1], 
            self.col_index, 
            self.MAX_SHIFT
            )
        if len(selected_cols) == 0:
            return 100000

        X_train = self.X_train_full[:, selected_cols]
        Y_train = self.y_train_full

        X_test = self.X_test_full[:, selected_cols]
        Y_test = self.y_test_full

        X_std_train = (X_train - self.mu[selected_cols]) / self.sigma[selected_cols]
        X_std_test = (X_test - self.mu[selected_cols]) / self.sigma[selected_cols]



        # Train model
        clf = LinearRegression()
        # Apply cross validation
        # clf.fit(X_std_train, Y_train)
        iav=np.std(Y_train) # inter-annual variablity of target
        score = np.abs(cross_val_score(clf, X_std_train, Y_train, cv=5,scoring='neg_root_mean_squared_error'))/iav
        score = score.mean()
        # score = cross_val_score(clf, X_std_train, Y_train, cv=5, scoring="f1").mean()

        # Save solution
        #history.append([score, Y_test, solution])
        clf.fit(X_std_train, Y_train)
        Y_pred = clf.predict(X_std_test)
        nrmse_test = root_mean_squared_error(Y_pred,Y_test)/iav
        print(score, nrmse_test)
        #print(score, f1_score(Y_pred,Y_test))

        self.fitness_cache[key] = score
        #self.fitness_cache[key] = 1/score
        #elapsed = time.perf_counter() - t0
        #print(f"objective time: {elapsed:.4f} s")
        sol_file = pd.concat([sol_file, pd.DataFrame({'CV': [score], 'Test': [nrmse_test], 'Sol': [solution]})], ignore_index=True)
        sol_file.to_csv(self.indiv_file,sep=' ',header=sol_file.columns,index=None)
    
        return score
    
    """
    This will be the function used to generate random vectorsfor the initializatio of the algorithm
    """
    def random_solution(self):
        return np.random.choice(self.sup_lim[0], self.size, replace=True)
    
    """
    This will be the function that will repair solutions, or in other words, makes a solution
    outside the domain of the function into a valid one.
    If this is not needed simply return "solution"
    """
    def repair_solution(self, solution):

        # unique = np.unique(solution)
        # if len(unique) < len(solution):
        #     pool = np.setdiff1d(np.arange(self.inf_lim[0], self.sup_lim[0]), unique)
        #     new = np.random.choice(pool, len(solution) - len(unique), replace=False)
        #     solution = np.concatenate((unique, new))
        return np.clip(solution, self.inf_lim, self.sup_lim)

