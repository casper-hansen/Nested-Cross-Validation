import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
import numpy as np
from sklearn.metrics import mean_squared_error

def nested_cv(X, y, model, params_grid, outer_kfolds, 
                       inner_kfolds, metric = mean_squared_error,
                       lower_score_is_better = True, sqrt_of_score = False, 
                       randomized_search = True, randomized_search_iter = 10):
    '''A general method to handle nested cross-validation for any estimator that
    implements the scikit-learn estimator interface.
    Parameters
    ----------
    X : pandas dataframe (rows, columns)
        Training dataframe, where rows is total number of observations and columns
        is total number of features
        
    y : pandas dataframe
        Output dataframe, also called output variable. y is what you want to predict.
        
    model : estimator
        The estimator implements scikit-learn estimator interface.
        
    params_grid : dict
        The dict contains hyperparameters for model.
        
    outer_kfolds : int
        Number of outer K-partitions in KFold
        
    inner_kfolds : int
        Number of inner K-partitions in KFold
        
    metric : callable from sklearn.metrics, default = mean_squared_error
        A scoring metric used to score each model
        
    lower_score_is_better : boolean, default = True
        Whether or not higher or lower is better for metric
        
    sqrt_of_score : boolean, default = False
        Whether or not if the square root should be taken of score
        
    randomized_search : boolean, default = True
        Whether to use gridsearch or randomizedsearch from sklearn
        
    randomized_search_iter : int, default = 10
        Number of iterations for randomized search
        
    Returns
    -------
    outer_scores
         Outer Score List.
         
    best_inner_score_list
         Best inner scores for each outer loop
         
    best_inner_params_list
         Best inner params for each outer loop
    '''
    print('\n{0} <-- Running this model now'.format(type(model).__name__))
    outer_cv = KFold(n_splits=outer_kfolds,shuffle=True)
    inner_cv = KFold(n_splits=inner_kfolds,shuffle=True)
    
    outer_scores = []
    variance = []
    best_inner_params_list = []
    best_inner_score_list = []
    
    
    # Split X and y into K-partitions
    for (i, (train_index,test_index)) in enumerate(outer_cv.split(X,y)):
        print('\n{0}/{1} <-- Current outer fold'.format(i+1,outer_kfolds))
        X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
        y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]
        inner_params = []
        inner_scores = []
        best_inner_params = {}
        best_inner_score = None
        
        # Split X_train_outer and y_train_outer into K-partitions
        for (j, (train_index_inner,test_index_inner)) in enumerate (inner_cv.split(X_train_outer,y_train_outer)):
            print('\n\t{0}/{1} <-- Current inner fold'.format(j+1,inner_kfolds))
            X_train_inner, X_test_inner = X_train_outer.iloc[train_index_inner], X_train_outer.iloc[test_index_inner]
            y_train_inner, y_test_inner = y_train_outer.iloc[train_index_inner], y_train_outer.iloc[test_index_inner]
            best_score = None
            best_grid = {}
            
            # Run either RandomizedSearch or GridSearch for input parameters
            for param_dict in ParameterSampler(param_distributions=params_grid,n_iter=randomized_search_iter) if randomized_search else ParameterGrid(param_grid=params_grid):
                # Set parameters, train model on inner split, predict results.
                model.set_params(**param_dict)
                model.fit(X_train_inner,y_train_inner)
                inner_pred = model.predict(X_test_inner)
                internal_grid_score = metric(y_test_inner,inner_pred)
                
                # Find best score and corresponding best grid
                if (best_score == None or internal_grid_score < best_score) and (lower_score_is_better):
                    if sqrt_of_score:
                        best_score = np.sqrt(internal_grid_score)
                    else:
                        best_score = internal_grid_score
                    best_grid = param_dict
                elif (best_score == None or internal_grid_score > best_score) and (not lower_score_is_better):
                    if sqrt_of_score:
                        best_score = np.sqrt(internal_grid_score)
                    else:
                        best_score = internal_grid_score
                    best_grid = param_dict
                            
            # Best grid and score found by the search
            inner_params.append(best_grid)
            inner_scores.append(best_score)
        
        # Look through all inner scores, select the best one
        for idx, score in enumerate(inner_scores):
            if (best_inner_score == None or score < best_inner_score) and (lower_score_is_better):
                best_inner_score = score
                best_inner_params = inner_params[idx]
            elif (best_inner_score == None or score > best_inner_score) and (not lower_score_is_better):
                best_inner_score = score
                best_inner_params = inner_params[idx]
        
        best_inner_params_list.append(best_inner_params)
        best_inner_score_list.append(best_inner_score)
        
        # Train model with best inner parameters on the outer split
        model.set_params(**best_inner_params)
        model.fit(X_train_outer,y_train_outer)
        pred = model.predict(X_test_outer)
        
        if sqrt_of_score:
            outer_scores.append(np.sqrt(metric(y_test_outer,pred)))
        else:
            outer_scores.append(metric(y_test_outer,pred))
        
        # Append variance
        variance.append(np.var(pred,ddof=1))
        print('\nResults for outer fold:\nBest inner parameters was: {0}'.format(best_inner_params_list[i]))
        print('Outer score: {0}'.format(outer_scores[i]))
        print('Inner score: {0}'.format(best_inner_score_list[i]))
    
    # Plot score vs variance
    plt.figure()
    plt.subplot(211)
    
    variance_plot, = plt.plot(variance,color='b')
    score_plot, = plt.plot(outer_scores, color='r')
    
    plt.legend([variance_plot, score_plot],
               ["Variance", "Score"],
               bbox_to_anchor=(0, .4, .5, 0))
    
    plt.title("{0}: Score VS Variance".format(type(model).__name__),
              x=.5, y=1.1, fontsize="15")
    
    return outer_scores, best_inner_score_list, best_inner_params_list