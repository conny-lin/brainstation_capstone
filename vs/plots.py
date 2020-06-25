# collectiof of plots
import numpy as np
import matplotlib.pyplot as plt

def hyperparameterplot(hyperparameter_list, train_score_list, test_score_list, \
                        hyperparameter_name='', xscale='linear',titlename=''):
    
    # check if hyperparameter list is string or number
    if isinstance(hyperparameter_list[0], str):
        hyperparameter_label = hyperparameter_list.copy()
        hyperparameter_list = range(len(hyperparameter_list))
    # graph
    plt.figure()
    plt.plot(hyperparameter_list, train_score_list, color='blue', label='train')
    plt.plot(hyperparameter_list, test_score_list, color='red', label='test')
    plt.title(titlename)
    plt.xlabel(hyperparameter_name)
    if isinstance(hyperparameter_list[0], str):
        plt.xticks(labels=hyperparameter_label)
    plt.ylabel('accuracy score')
    plt.xscale(xscale)
    plt.legend()
    plt.show()

def gridcvplot(hyperparameter_list, means, stds, hyperparameter_name='', titlename=''):
    plt.figure()
    plt.errorbar(hyperparameter_list, means, stds, color='blue', label='train')
    plt.title(titlename)
    plt.xlabel(hyperparameter_name)
    plt.ylabel('mean accuracy score')
    plt.show()

def heatmap_corr(cor):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # create mask
    i = cor<0
    corr_positive = cor.copy()
    corr_positive[i] = cor[i]*-1
    corr_positive[~i] = cor[~i]*1
    mask = np.triu(np.ones_like(corr_positive, dtype=np.bool))
    # plot figure
    plt.figure()
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cor, square=True,mask=mask,cmap=cmap,
                linewidths=.5, ax=ax)
    plt.title('rho')
    plt.show()

def plothist_2groups(X_no, X_etoh, bins, feature_name):
    plt.hist(X_no, bins, alpha=0.5, label='no ethanol')
    plt.hist(X_etoh, bins, alpha=0.5, label='ethanol')
    plt.title(feature_name)
    plt.legend(loc='upper right')
    plt.show()
