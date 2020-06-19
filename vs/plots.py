# collectiof of plots

def hyperparameterplot(hyperparameter_list, train_score_list, test_score_list, \
                        hyperparamter_name='', titlename='')
    plt.figure()
    plt.plot(hyperparameter_list, train_score_list, color='blue', label='train')
    plt.plot(hyperparameter_list, test_score_list, color='red', label='test')
    plt.title(titlename)
    plt.xlabel(hyperparamter_name)
    plt.ylabel('score')
    plt.legend()
    plt.show()