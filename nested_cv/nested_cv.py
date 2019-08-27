import logging as log
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.utils.multiclass import type_of_target
from joblib import Parallel, delayed

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
    n_jobs : int
        Number of jobs to run in parallel

    cv_options: dict, default = {}
        Nested Cross-Validation Options, check docs for details.

        metric : callable from sklearn.metrics, default = mean_squared_error
            A scoring metric used to score each model

        metric_score_indicator_lower : boolean, default = True
            Choose whether lower score is better for the metric calculation or higher score is better,
            `True` means lower score is better.

        sqrt_of_score : boolean, default = False
            Whether or not the square root should be taken of score

        randomized_search : boolean, default = False
            Whether to use gridsearch or randomizedsearch from sklearn

        randomized_search_iter : int, default = 10
            Number of iterations for randomized search

        recursive_feature_elimination : boolean, default = False
            Whether to do recursive feature selection (rfe) for each set of different hyperparameters
            in the inner most loop of the fit function.

        rfe_n_features : int, default = 1
            If recursive_feature_elimination is enabled, select n number of features
        
        predict_proba : boolean, default = False
            If true, predict probabilities instead for a class, instead of predicting a class
        
        multiclass_average : string, default = 'binary'
            For some classification metrics with a multiclass prediction, you need to specify an
            average other than 'binary'
    '''

    def __init__(self, model, params_grid, outer_kfolds, inner_kfolds, n_jobs = 1, cv_options={}):
        self.model = model
        self.params_grid = params_grid
        self.outer_kfolds = outer_kfolds
        self.inner_kfolds = inner_kfolds
        self.n_jobs = n_jobs
        self.metric = cv_options.get('metric', mean_squared_error)
        self.metric_score_indicator_lower = cv_options.get(
            'metric_score_indicator_lower', True)
        self.sqrt_of_score = cv_options.get('sqrt_of_score', False)
        self.randomized_search = cv_options.get('randomized_search', False)
        self.randomized_search_iter = cv_options.get(
            'randomized_search_iter', 10)
        self.recursive_feature_elimination = cv_options.get(
            'recursive_feature_elimination', False)
        self.rfe_n_features = cv_options.get(
            'rfe_n_features', 0)
        self.predict_proba = cv_options.get(
            'predict_proba', False)
        self.multiclass_average = cv_options.get(
            'multiclass_average', 'binary')
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

    # a function to handle recursive feature elimination
    def _fit_recursive_feature_elimination(self, X_train_outer, y_train_outer, X_test_outer):
        rfe = RFECV(estimator=self.model,
                    min_features_to_select=self.rfe_n_features, cv=self.inner_kfolds, n_jobs = self.n_jobs)
        rfe.fit(X_train_outer, y_train_outer)
        
        log.info('Best number of features was: {0}'.format(rfe.n_features_))

        # Assign selected features to data
        return rfe.transform(X_train_outer), rfe.transform(X_test_outer)
    
    def _predict_and_score(self, X_test, y_test):
        #XXX: Implement type_of_target(y)
        
        if(self.predict_proba):
            y_type = type_of_target(y_test)
            if(y_type in ('binary')):
                pred = self.model.predict_proba(X_test)[:,1]
            else:
                pred = self.model.predict_proba(X_test)
                
        else:
            pred = self.model.predict(X_test)
        
        if(self.multiclass_average == 'binary'):
            return self.metric(y_test, pred), pred
        else:
            return self.metric(y_test, pred, average=self.multiclass_average), pred
    def _best_of_results(self, results):
        best_score = None
        best_parameters = {}
        
        for score_parameter in results:
            if(self.metric_score_indicator_lower):
                if(best_score == None or score_parameter[0] < best_score):
                    best_score = score_parameter[0]
                    best_parameters = score_parameter[1]
            else:
                if(best_score == None or score_parameter[0] > best_score):
                    best_score = score_parameter[0]
                    best_parameters = score_parameter[1]
        
        return best_score, best_parameters

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
        
        log.debug(
            '\n{0} <-- Running this model now'.format(type(self.model).__name__))

        self.X = X
        self.y = y

        # If Pandas dataframe or series, convert to array
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        if(self.randomized_search):
            param_func = ParameterSampler(param_distributions=self.params_grid,
                                                   n_iter=self.randomized_search_iter)
        else:
            param_func = ParameterGrid(param_grid=self.params_grid)
        
        outer_cv = KFold(n_splits=self.outer_kfolds, shuffle=True)
        inner_cv = KFold(n_splits=self.inner_kfolds, shuffle=True)

        outer_scores = []
        variance = []
        best_inner_params_list = []  # Change both to by one thing out of key-value pair
        best_inner_score_list = []

        # Split X and y into K-partitions to Outer CV
        for (i, (train_index, test_index)) in enumerate(outer_cv.split(X, y)):
            log.debug(
                '\n{0}/{1} <-- Current outer fold'.format(i+1, self.outer_kfolds))
            X_train_outer, X_test_outer = X[train_index], X[test_index]
            y_train_outer, y_test_outer = y[train_index], y[test_index]
            best_inner_params = {}
            best_inner_score = None
            search_scores = []

            # Split X_train_outer and y_train_outer into K-partitions to be inner CV
            for (j, (train_index_inner, test_index_inner)) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
                log.debug(
                    '\n\t{0}/{1} <-- Current inner fold'.format(j+1, self.inner_kfolds))
                X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
                y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]
                
                if self.recursive_feature_elimination:
                        X_train_inner, X_test_inner = self._fit_recursive_feature_elimination(
                                    X_train_inner, y_train_inner, X_test_inner)
                
                def _parallel_fitting(X_train_inner, X_test_inner, y_train_inner, y_test_inner, param_dict):
                    log.debug(
                        '\n\tFitting these parameters:\n\t{0}'.format(param_dict))
                    # Set hyperparameters, train model on inner split, predict results.
                    self.model.set_params(**param_dict)

                    # Fit model with current hyperparameters and score it
                    self.model.fit(X_train_inner, y_train_inner)
                    
                    # Predict and score model
                    inner_grid_score, inner_pred = self._predict_and_score(X_test_inner, y_test_inner)
                    
                    # Cleanup for Keras
                    if(type(self.model).__name__ == 'KerasRegressor' or
                       type(self.model).__name__ == 'KerasClassifier'):
                        from keras import backend as K
                        K.clear_session()
                    
                    return self._transform_score_format(inner_grid_score), param_dict
            
                results = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_fitting)(
                                                    X_train_inner, X_test_inner,
                                                    y_train_inner, y_test_inner,
                                                    param_dict=parameters)
                                            for parameters in param_func)
                search_scores.extend(results)
            
            best_inner_score, best_inner_params = self._best_of_results(search_scores)
            
            best_inner_params_list.append(best_inner_params)
            best_inner_score_list.append(best_inner_score)

            # Fit the best hyperparameters from one of the K inner loops
            self.model.set_params(**best_inner_params)
            self.model.fit(X_train_outer, y_train_outer)
            
            # Get score and prediction
            score,pred = self._predict_and_score(X_test_outer, y_test_outer)
            outer_scores.append(self._transform_score_format(score))

            # Append variance
            variance.append(np.var(pred, ddof=1))

            log.debug('\nResults for outer fold:\nBest inner parameters was: {0}'.format(
                best_inner_params_list[i]))
            log.debug('Outer score: {0}'.format(outer_scores[i]))
            log.debug('Inner score: {0}'.format(best_inner_score_list[i]))
         
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
