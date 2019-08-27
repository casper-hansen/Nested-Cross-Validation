#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:14:06 2019

@author: casperbogeskovhansen
"""

from nested_cv import NestedCV

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# When using Random Search, we get a user warning with this little number of hyperparameters
# Suppress it
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

boston = load_boston()
X = boston.data
y = boston.target

# Define a parameters grid
param_grid = {
     'max_depth': [3, None],
     'n_estimators': [10]
}

NCV = NestedCV(model=RandomForestRegressor(), params_grid=param_grid, outer_kfolds=5, inner_kfolds=5,
               cv_options={'sqrt_of_score':True, 'randomized_search_iter':30,
                           'recursive_feature_elimination':False, 'rfe_n_features':2})
NCV.fit(X=X,y=y)

print(NCV.outer_scores)