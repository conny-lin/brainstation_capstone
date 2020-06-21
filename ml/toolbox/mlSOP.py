import numpy as np
import time
from sklearn.model_selection import GridSearchCV


class test_model:
    def __init__(self):
        # set test and train score
        self.test_acc = []
        self.train_acc = []
    
    def score_data(self, model, datadict):
        # get input
        self.model = model
        self.data = datadict
        # fit model
        self.model.fit(self.data['X_train'], self.data['y_train'])
        # train score
        train_score = self.model.score(self.data['X_train'], self.data['y_train'])
        print(f'\tTrain Score: \t\t{train_score}')
        self.train_acc.append(train_score)
        # test score
        test_score = self.model.score(self.data['X_test'], self.data['y_test'])
        print(f'\tTest Score: \t\t{test_score}')
        self.test_acc.append(test_score)
        # print comparison
        print(f'\tOverfit (train - test): \t{train_score - test_score}')


def test_train_score_capture(model, data, train_acc, test_acc):
    # fit model
    model.fit(data['X_train'], data['y_train'])
    # train score
    train_score = model.score(data['X_train'], data['y_train'])
    print(f"\tTrain Score: {train_score}")
    train_acc.append(train_score)
    # test score
    test_score = model.score(data['X_test'], data['y_test'])
    print(f"\tTest Score: {test_score}")
    test_acc.append(test_score)
    return train_acc, test_acc


class ml_timer:
    def __init__(self):
        # initate session start time
        self.start = time.time()
        # initiate holder for times
        self.session_times = []
        print('timer starts')

    def param_start(self):
        # update current session start time
        self.current_session_start = time.time()

    def param_end(self):
        end_time = time.time()
        elapsed_time = end_time - self.current_session_start
        print(f'\telapsed time {elapsed_time/60:.3f} min')
        self.session_times.append(elapsed_time)
    
    def session_end(self):
        self.end = time.time()
    
    def get_time(self):
        print(f'total time: {(self.end - self.start)/60:.3f} min')
        return self.session_times
    

class GridSearchCVHelper:
    def __init__(self, model, grid):
        self.grid = grid
        self.model = model
    
    def run(self, X, y, **kwargs):
        # process kwargs
        n_jobs = kwargs.pop('n_jobs', -1)
        cv = kwargs.pop('cv', 3)
        scoring = kwargs.pop('scoring', 'accuracy')
        error_score = kwargs.pop('error_score', 0)
        # instantiate grid search
        self.grid_search = GridSearchCV(estimator=self.model, 
                                param_grid=self.grid,
                                n_jobs=n_jobs, 
                                cv=cv, scoring=scoring, 
                                error_score=error_score)
        self.grid_result = self.grid_search.fit(X, y)
        return self.grid_result, self.grid_search
    
    def print_result(self):
        print("Best: %f using %s" % (self.grid_result.best_score_, 
                                    self.grid_result.best_params_))
        means = self.grid_result.cv_results_['mean_test_score']
        stds = self.grid_result.cv_results_['std_test_score']
        params = self.grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        return means, stds, params
    
    