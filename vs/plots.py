# collectiof of plots
import numpy as np
import matplotlib.pyplot as plt

def hyperparameterplot(hyperparameter_list, train_score_list, test_score_list, \
                        hyperparameter_name='', titlename=''):
    plt.figure()
    plt.plot(hyperparameter_list, train_score_list, color='blue', label='train')
    plt.plot(hyperparameter_list, test_score_list, color='red', label='test')
    plt.title(titlename)
    plt.xlabel(hyperparameter_name)
    plt.ylabel('score')
    plt.legend()
    plt.show()

def gridcvplot(hyperparameter_list, means, stds, hyperparameter_name='', titlename=''):
    plt.figure()
    plt.errorbar(hyperparameter_list, means, stds, color='blue', label='train')
    plt.title(titlename)
    plt.xlabel(hyperparameter_name)
    plt.ylabel('mean accuracy score')
    plt.show()