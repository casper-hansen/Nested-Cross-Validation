import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE, RFECV


class NestedCV():
    '''A general class to handle nested cross-validation for any estimator that
    implements the scikit-learn estimator interface.

    Parameters
    ----------
    model : estimator
        The estimator implements scikit-learn estimator interface.

    params_grid : dict
        The dict contains hyperparameters for model.

    outer_kfolds : int
        Number of outer K-partitions in KFold

    inner_kfolds : int
        Number of inner K-partitions in KFold

    cv_options: dict, default = {}
        Nested Cross-Validation Options, check docs for details.

        metric : callable from sklearn.metrics, default = mean_squared_error
            A scoring metric used to score each model

        metric_score_indicator_lower : boolean, default = True
            Choose whether lower score is better for the metric calculation or higher score is better,
            `True` means lower score is better.

        sqrt_of_score : boolean, default = False
            Whether or not the square root should be taken of score

        randomized_search : boolean, default = True
            Whether to use gridsearch or randomizedsearch from sklearn

        randomized_search_iter : int, default = 10
            Number of iterations for randomized search

        recursive_feature_elimination : boolean, default = False
            Whether to do feature elimination
    '''

    def __init__(self, model, params_grid, outer_kfolds, inner_kfolds, cv_options={}):
        self.model = model
        self.params_grid = params_grid
        self.outer_kfolds = outer_kfolds
        self.inner_kfolds = inner_kfolds
        self.metric = cv_options.get('metric', mean_squared_error)
        self.metric_score_indicator_lower = cv_options.get(
            'metric_score_indicator_lower', True)
        self.sqrt_of_score = cv_options.get('sqrt_of_score', False)
        self.randomized_search = cv_options.get('randomized_search', True)
        self.randomized_search_iter = cv_options.get(
            'randomized_search_iter', 10)
        self.recursive_feature_elimination = cv_options.get(
            'recursive_feature_elimination', False)
        self.outer_scores = []
        self.best_params = {}
        self.best_inner_score_list = []
        self.variance = []

    # to check if use sqrt_of_score and handle the different cases
    def _transform_score_format(self, scoreValue):
        if self.sqrt_of_score:
            return np.sqrt(scoreValue)
        return scoreValue

    # to convert array of dict to dict with array values, so it can be used as params for parameter tuning
    def _score_to_best_params(self, best_inner_params_list):
        params_dict = {}
        for best_inner_params in best_inner_params_list:
            for key, value in best_inner_params.items():
                if key in params_dict:
                    if value not in params_dict[key]:
                        params_dict[key].append(value)
                else:
                    params_dict[key] = [value]
        return params_dict

    # a method to handle  recursive feature elimination
    def _fit_recursive_feature_elimination(self, best_inner_params, X_train_outer, y_train_outer, X_test_outer):
        print('\nRunning recursive feature elimination for outer loop... (SLOW)')
        # K-fold (inner_kfolds) recursive feature elimination
        rfe = RFECV(estimator=self.model, min_features_to_select=20,
                    scoring='neg_mean_squared_error', cv=self.inner_kfolds, n_jobs=-1)
        rfe.fit(X_train_outer, y_train_outer)

        # Assign selected features to data
        print('Best number of features was: {0}'.format(rfe.n_features_))
        X_train_outer_rfe = rfe.transform(X_train_outer)
        X_test_outer_rfe = rfe.transform(X_test_outer)

        # Train model with best inner parameters on the outer split
        self.model.set_params(**best_inner_params)
        self.model.fit(X_train_outer_rfe, y_train_outer)
        return self.model.predict(X_test_outer_rfe)

    def fit(self, X, y):
        '''A method to fit nested cross-validation 
        Parameters
        ----------
        X : pandas dataframe (rows, columns)
            Training dataframe, where rows is total number of observations and columns
            is total number of features
    
        y : pandas dataframe
            Output dataframe, also called output variable. y is what you want to predict.
    
        Returns
        -------
        It will not return directly the values, but it's accessable from the class object it self.
        You should be able to access:
        
        variance
            Model variance by numpy.var()
            
        outer_scores 
            Outer score List.
            
        best_inner_score_list 
            Best inner scores for each outer loop
            
        best_params 
            All best params from each inner loop cumulated in a dict
            
        best_inner_params_list 
            Best inner params for each outer loop as an array of dictionaries
        '''
    
        print('\n{0} <-- Running this model now'.format(type(self.model).__name__))
        outer_cv = KFold(n_splits=self.outer_kfolds, shuffle=True)
        inner_cv = KFold(n_splits=self.inner_kfolds, shuffle=True)
        model = self.model
    
        outer_scores = []
        variance = []
        best_inner_params_list = []  # Change both to by one thing out of key-value pair
        best_inner_score_list = []
    
        # Split X and y into K-partitions to Outer CV
        for (i, (train_index, test_index)) in enumerate(outer_cv.split(X, y)):
            print('\n{0}/{1} <-- Current outer fold'.format(i+1, self.outer_kfolds))
            X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
            y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]
            best_inner_params = {}
            best_inner_score = None
    
            # Split X_train_outer and y_train_outer into K-partitions to be inner CV
            for (j, (train_index_inner, test_index_inner)) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
                print('\n\t{0}/{1} <-- Current inner fold'.format(j+1, self.inner_kfolds))
                X_train_inner, X_test_inner = X_train_outer.iloc[
                    train_index_inner], X_train_outer.iloc[test_index_inner]
                y_train_inner, y_test_inner = y_train_outer.iloc[
                    train_index_inner], y_train_outer.iloc[test_index_inner]
    
                # Run either RandomizedSearch or GridSearch for input parameters
                for param_dict in ParameterSampler(param_distributions=self.params_grid, 
                                                   n_iter=self.randomized_search_iter) if (self.randomized_search) else (
                                                           ParameterGrid(param_grid=self.params_grid)):
                    # Set parameters, train model on inner split, predict results.
                    model.set_params(**param_dict)
                    model.fit(X_train_inner, y_train_inner)
                    inner_pred = model.predict(X_test_inner)
                    inner_grid_score = self.metric(y_test_inner, inner_pred)
                    current_inner_score_value = best_inner_score
    
                    # Find best score and corresponding best grid
                    if(best_inner_score is not None):
                        if(self.metric_score_indicator_lower and best_inner_score > inner_grid_score):
                            best_inner_score = self._transform_score_format(inner_grid_score)
                            
                        elif (not self.metric_score_indicator_lower and best_inner_score < inner_grid_score):
                            best_inner_score = self._transform_score_format(inner_grid_score)
                    else:
                        best_inner_score = self._transform_score_format(inner_grid_score)
                        current_inner_score_value = best_inner_score+1  # first time random thing
                        
                    # Update best_inner_grid once rather than calling it under each if statement
                    if(current_inner_score_value is not None and current_inner_score_value != best_inner_score):
                        best_inner_params = param_dict
    
            best_inner_params_list.append(best_inner_params)
            best_inner_score_list.append(best_inner_score)
    
            if self.recursive_feature_elimination:
                pred = self._fit_recursive_feature_elimination(
                    best_inner_params, X_train_outer, y_train_outer, X_test_outer)
            else:
                # Train model with best inner parameters on the outer split
                model.set_params(**best_inner_params)
                model.fit(X_train_outer, y_train_outer)
                pred = model.predict(X_test_outer)
    
            outer_scores.append(self._transform_score_format(
                self.metric(y_test_outer, pred)))
    
            # Append variance
            variance.append(np.var(pred, ddof=1))
    
            print('\nResults for outer fold:\nBest inner parameters was: {0}'.format(
                best_inner_params_list[i]))
            print('Outer score: {0}'.format(outer_scores[i]))
            print('Inner score: {0}'.format(best_inner_score_list[i]))
    
        self.variance = variance
        self.outer_scores = outer_scores
        self.best_inner_score_list = best_inner_score_list
        self.best_params = self._score_to_best_params(best_inner_params_list)
        self.best_inner_params_list = best_inner_params_list

    # Method to show score vs variance chart. You can run it only after fitting the model.
    def score_vs_variance_plot(self):
        # Plot score vs variance
        plt.figure()
        plt.subplot(211)

        variance_plot, = plt.plot(self.variance, color='b')
        score_plot, = plt.plot(self.outer_scores, color='r')

        plt.legend([variance_plot, score_plot],
                   ["Variance", "Score"],
                   bbox_to_anchor=(0, .4, .5, 0))

        plt.title("{0}: Score VS Variance".format(type(self.model).__name__),
                  x=.5, y=1.1, fontsize="15")
