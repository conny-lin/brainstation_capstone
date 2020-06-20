
from sklearn.model_selection import GridSearchCV

# primt GridSearchCV Results
def print_summary(grid_result):
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print('printing individual results')
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return means, stds, params

def grid_tuning(model, grid, **kwargs):
    # get kwargs
    random_state = kwargs.pop('random_state', 318)
    n_jobs = kwargs.pop('n_jobs', -1)
    cv = kwargs.pop('cv', 5)
    scoring = kwargs.pop('scoring', 'accuracy'')
    error_score = kwargs.pop('error_score', 0)
    verbose = kwargs.pop('verbose', 3)
    # run search
    grid_search = GridSearchCV(estimator=model, param_grid=grid, 
                                n_jobs=-1, 
                                cv=cv, 
                                scoring='accuracy',
                                error_score=0, 
                                verbose=3)
    grid_result = grid_search.fit(X_train, y_train)
    # get results
    means, stds, params = print_summary(grid_result)
    return means, stds, params
