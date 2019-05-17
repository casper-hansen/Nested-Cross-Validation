# Nested-Cross-Validation
This repository implements a general nested cross-validation function. Ready to use with ANY estimator that implements the Scikit-Learn estimator interface.

## Usage
Be mindful of the options that are available for NestedCV. Some cross-validation options are defined in a dictionary `cv_options`:

### Single algorithm
Here is a single example using Random Forest
```python
# Define a parameters grid
param_grid = {
     'max_depth': [3, None],
     'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
     'max_features' : [50,100,150,200] # Note: You might not have that many features
}

# Define parameters for function
# Default scoring: RMSE
nested_CV_search = NestedCV(model=RandomForestRegressor(), params_grid=param_grid , outer_kfolds=5, inner_kfolds=5, 
                      	    cv_options={'sqrt_of_score':True, 'randomized_search_iter':30})
nested_CV_search.fit(X=X,y=y)
print('\nCumulated best parameter grid was:\n{0}'.format(nested_CV_search.best_params))
```

### Multiple algorithms
Here is an example using Random Forest, XGBoost and LightGBM.
```python
models_to_run = [RandomForestRegressor(), xgb.XGBRegressor(), lgb.LGBMRegressor()]
models_param_grid = [ 
                    { # 1st param grid, corresponding to RandomForestRegressor
                            'max_depth': [3, None],
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'max_features' : [50,100,150,200]
                    }, 
                    { # 2nd param grid, corresponding to XGBRegressor
                            'learning_rate': [0.05],
                            'colsample_bytree': np.linspace(0.3, 0.5),
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'reg_alpha' : (1,1.2),
                            'reg_lambda' : (1,1.2,1.4)
                    },
                    { # 3rd param grid, corresponding to LGBMRegressor
                            'learning_rate': [0.05],
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'reg_alpha' : (1,1.2),
                            'reg_lambda' : (1,1.2,1.4)
                    }
                    ]

for i,model in enumerate(models_to_run):
    nested_CV_search = NestedCV(model=model, params_grid=models_param_grid[i], outer_kfolds=5, inner_kfolds=5, 
                      cv_options={'sqrt_of_score':True, 'randomized_search_iter':30})
    nested_CV_search.fit(X=X,y=y)
    model_param_grid = nested_CV_search.best_params

    print('\nCumulated best parameter grid was:\n{0}'.format(model_param_grid))
```

### `cv_options` documentation
**`metric` :** Callable from sklearn.metrics, default = mean_squared_error

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A scoring metric used to score each model

**`metric_score_indicator_lower` :** boolean, default = True

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Choose whether lower score is better for the metric calculation or higher score is better.

**`sqrt_of_score` :** boolean, default = False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether or not if the square root should be taken of score

**`randomized_search` :** boolean, default = True

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use gridsearch or randomizedsearch from sklearn

**`randomized_search_iter` :** int, default = 10

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of iterations for randomized search

**`recursive_feature_elimination` :** boolean, default = False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to do feature elimination

## How to use the output?
We suggest using the best parameters from the best outer score with your full data in a GridSearch Cross-Validation. They can be accessed on the NestedCV object by `.best_params`

## Limitations
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) implements a `early_stopping_rounds`, which cannot be used in this implementation. Other similar parameters might not work in combination with this implementation. The function will have to be adopted to use special parameters like that.
- The function only works with [Pandas dataframes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), and does currently not support NumPy arrays.
- Limited feature selection/elimination included (only executed after inner loop has run)

## What did we learn?
- Using [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) will lead to a faster implementation, since the Scikit-Learn community has implemented many functions that does much of the work.

## Why use Nested Cross-Validation?
Controlling the bias-variance tradeoff is an essential and important in machine learning, indicated by [[Cawley and Talbot, 2010]](http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf). Many articles indicate that this is possible by the use of nested cross-validation, one of them by [Varma and Simon, 2006](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1397873/pdf/1471-2105-7-91.pdf). It has many applications and has many applications. Other interesting literature for nested cross-validation are [[Varoquaox et al., 2017]](https://arxiv.org/pdf/1606.05201.pdf) and [[Krstajic et al., 2014]](https://jcheminf.biomedcentral.com/track/pdf/10.1186/1758-2946-6-10).
