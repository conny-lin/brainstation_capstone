# collectio of LogisticRegression codes
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression


def quicktest(Nutcracker, **kwargs):
    '''
    intake Nutcracker object and perform quick tests on semi-optimized parameters
    '''
    kwargs_dict = kwargs
    # ge kwargs and define defaults
    C = kwargs.pop('C', 0.05)
    random_state = kwargs.pop('random_state', 318)
    max_iter = kwargs.pop('max_iter', 1000)
    print_accuracy = kwargs.pop('print_accuracy', True)

    # run logistic regression
    print('running logistic regression')
    OLS = LogisticRegression(C=C, random_state=random_state, max_iter=max_iter)
    # fit
    OLS.fit(Nutcracker.X_train_scaled, Nutcracker.y_train)
    # get train score
    train_score = OLS.score(Nutcracker.X_train_scaled, Nutcracker.y_train)
    test_score = OLS.score(Nutcracker.X_test_scaled, Nutcracker.y_test)
    # prin accuracy score
    if print_accuracy:
        print(f'train score: {train_score}')
        print(f'test score: {test_score}')
    return train_score, test_score, kwargs_dict
