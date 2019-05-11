# Nested-Cross-Validation
This repository implements a general nested cross-validation function. Ready to use with ANY estimator that implements the Scikit-Learn estimator interface.

# Usage (Example)
Here is an example using Random Forest, XGBoost and LightGBM.
```python
models_to_run = [RandomForestRegressor(), xgb.XGBRegressor(), lgb.LGBMRegressor()]
models_param_grid = [ 
                    { # 1st param grid, corresponding to RandomForestRegressor
                            'max_depth': [3, None],
                            'n_estimators': np.random.randint(100,1000,20)
                    }, 
                    { # 2nd param grid, corresponding to XGBRegressor
                            'colsample_bytree': np.linspace(0.3, 0.5),
                            'n_estimators': np.random.randint(100,1000,20)
                    },
                    { # 3rd param grid, corresponding to LGBMRegressor
                            'learning_rate': [0.05],
                            'n_estimators': np.random.randint(100,1000,20),
                            'num_leaves': np.random.randint(10,30,10),
                            'reg_alpha' : (1,1.2),
                            'reg_lambda' : (1,1.2,1.4)
                    }
                    ]

# Returns scores of RMSE. Remove sqrt_of_score for MSE.
returnarray = nested_cv(X=X, y=y, models=models_to_run, params_grid=models_param_grid,
                       outer_kfolds=5, inner_kfolds=5, sqrt_of_score = True, 
                       randomized_search_iter = 20)
```

# Limitations
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) implements a `early_stopping_rounds`, which cannot be used in this implementation. Other similar parameters might not work in combination with this implementation. The function will have to be adopted to use special parameters like that.
- The function only works with [Pandas dataframes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), and does currently not support NumPy arrays.
- No feature selection within the nested cross-validation.

# What did we learn?
- Using [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) will lead to a faster implementation, since the Scikit-Learn community has implemented many functions that does much of the work.

# Why use Nested Cross-Validation?
Controlling the bias-variance tradeoff is an essential and important in machine learning, indicated by [[Cawley and Talbot, 2010]](http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf). Many articles indicate that this is possible by the use of nested cross-validation, one of them by [Varma and Simon, 2006](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1397873/pdf/1471-2105-7-91.pdf). It has many applications and has many applications. Other interesting literature for nested cross-validation are [[Varoquaox et al., 2017]](https://arxiv.org/pdf/1606.05201.pdf) and [[Krstajic et al., 2014]](https://jcheminf.biomedcentral.com/track/pdf/10.1186/1758-2946-6-10).
