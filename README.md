# Nested-Cross-Validation
This repository implements a general nested cross-validation function. Ready to use with ANY estimator that implements the Scikit-Learn estimator interface.
## Installing the pacakge:
You can find the package on [pypi](https://pypi.org/project/nested-cv/)* and install it via pip by using the following command:
```bash
pip install nested-cv
```
You can also install it from the wheel file on the [Releases](https://github.com/casperbh96/Nested-Cross-Validation/releases) page.

\* we gradually push updates, pull this master from github if you want the absolute latest changes.

## Usage
Be mindful of the options that are available for NestedCV. Some cross-validation options are defined in a dictionary `cv_options`.
This package is optimized for any estimator that implements a scikit-learn wrapper, e.g. XGBoost, LightGBM, KerasRegressor, KerasClassifier etc.

### Single algorithm
Here is a single example using Random Forest
```python
#import the package 
from nested_cv import NestedCV
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
grid_nested_cv.score_vs_variance_plot()
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
### NestedCV Parameters 
| Name        | type           | description  |
| :------------- |:-------------| :-----|
| model      | estimator | The estimator implements scikit-learn estimator interface. |
| params_grid      | dictionary "dict"      |   The dict contains hyperparameters for model. |
| outer_kfolds | int      |    Number of outer K-partitions in KFold |
| inner_kfolds | int      | Number of inner K-partitions in KFold    | 
| cv_options | dictionary "dict"      |    [Next section](#cv_options-value-options) |
| n_jobs | int      | Number of jobs for joblib to run (multiprocessing)    | 

### `cv_options` value options
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

**`predict_proba` :** boolean, default = False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If true, predict probabilities instead for a class, instead of predicting a class

**`multiclass_average` :** string, default = 'binary'

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For some classification metrics with a multiclass prediction, you need to specify an
            average other than 'binary'

### Returns
**`variance` :** Model variance by numpy.var()

**`outer_scores` :** A list of the outer scores, from the outer cross-validation

**`best_inner_score_list` :** A list of best inner scores for each outer loop

**`best_params` :** All best params from each inner loop cumulated in a dict

**`best_inner_params_list` :** Best inner params for each outer loop as an array of dictionaries

## How to use the output?
We suggest looking at the best hyperparameters together with the score for each outer loop. Look at how stable the model appears to be in a nested cross-validation setting. If the outer score changes a lot, then it might indicate instability in your model. In that case, start over with making a new model.

### After Nested Cross-Validation?
If the results from nested cross-validation are stable: Run a normal cross-validation with the same procedure as in nested cross-validation, i.e. if you used feature selection in nested cross-validation, you should also do that in normal cross-validation.

## Limitations
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) implements a `early_stopping_rounds`, which cannot be used in this implementation. Other similar parameters might not work in combination with this implementation. The function will have to be adopted to use special parameters like that.

## What did we learn?
- Using [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) will lead to a faster implementation, since the Scikit-Learn community has implemented many functions that does much of the work.
- We have learned and applied this package in our main project about [House Price Prediction](https://github.com/casperbh96/house-price-prediction).

## Why use Nested Cross-Validation?
Controlling the bias-variance tradeoff is an essential and important task in machine learning, indicated by [[Cawley and Talbot, 2010]](http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf). Many articles indicate that this is possible by the use of nested cross-validation, one of them by [Varma and Simon, 2006](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1397873/pdf/1471-2105-7-91.pdf). Other interesting literature for nested cross-validation are [[Varoquaox et al., 2017]](https://arxiv.org/pdf/1606.05201.pdf) and [[Krstajic et al., 2014]](https://jcheminf.biomedcentral.com/track/pdf/10.1186/1758-2946-6-10).
