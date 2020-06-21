import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt



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
    


class GridSearchCVSOP:
    def __init__(self, model, grid):
        self.grid = grid
        self.model = model
    
    def run(self, X, y, **kwargs):
        # process kwargs
        n_jobs = kwargs.pop('n_jobs', -1)
        cv = kwargs.pop('cv', 3)
        scoring = kwargs.pop('scoring', 'accuracy')
        error_score = kwargs.pop('error_score', 0)
        verbose = kwargs.pop('verbose', 3)
        # instantiate grid search
        self.grid_search = GridSearchCV(estimator=self.model, 
                                param_grid=self.grid,
                                n_jobs=n_jobs, verbose=verbose,
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



class ModelEvaluation:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def cross_val_score(self, cv=5):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, 
                    self.data['X_train'], 
                    self.data['y_train'], 
                    cv=cv)
        print(f'cross validation scores: {scores}')
        print(f'validation score (mean):{np.mean(scores)}')
        print(f'validation score (std):{np.std(scores)}')
        self.cross_val_score_ = scores
        return self.cross_val_score_
    
    def fitmodel(self):
        self.model.fit(self.data['X_train'], self.data['y_train'])
        return self.model
    
    def predict(self):
        self.y_pred_test = self.model.predict(self.data['X_test'])
        self.y_pred_train = self.model.predict(self.data['X_train'])

    def accuracy_score(self):
        self.score_train = self.model.score(self.data["X_train"], self.data['y_train'])
        print(f'accuracy score on train: {self.score_train}')
        self.score_test = self.model.score(self.data['X_test'], self.data['y_test'])
        print(f'accuracy score on test: {self.score_test}')
        return self.score_train, self.score_test

    def confusion_matrix(self):
        # fitmodel and predict must proceed this.
        # define dataframe labels
        columns = ['Predicted normal', 'Predicted alcohol']
        indexname = ['True normal', 'True alcohol']
        from sklearn.metrics import confusion_matrix
        # run confusion matrix - test
        self.conf_matrix_test = confusion_matrix(self.data['y_test'], self.y_pred_test, 
                                            normalize='true')
        conf_matrix_test_df = pd.DataFrame(self.conf_matrix_test, columns=columns)
        conf_matrix_test_df.index = indexname
        print('\nconfusion matrix: test data')
        print(conf_matrix_test_df)
        # run confusion matrix - train
        self.conf_matrix_train = confusion_matrix(self.data['y_train'], 
                                            self.y_pred_train, 
                                            normalize='true')
        conf_matrix_train_df = pd.DataFrame(self.conf_matrix_train, columns=columns)
        conf_matrix_train_df.index = indexname
        print('\nconfusion matrix: train data')
        print(conf_matrix_train_df)
        return self.conf_matrix_test, self.conf_matrix_train
    
    def display_confusion_matrix(self):
        # confusion_matrix must proceed this
        display_labels = ['normal', 'alcohol']
        from sklearn.metrics import ConfusionMatrixDisplay
        print('\nconfusion matrix for test')
        plt.figure()
        ConfusionMatrixDisplay(self.conf_matrix_test, display_labels=display_labels).plot()
        plt.show()
        print('\nconfusion matrix for train')
        plt.figure()
        ConfusionMatrixDisplay(self.conf_matrix_train, display_labels=display_labels).plot()
        plt.show()

    def classification_report(self):
        from sklearn.metrics import classification_report
        self.eval_score_report = classification_report(self.data['y_test'], self.y_pred_test)
        print(self.eval_score_report)
    
    def print_evaluation_scores(self):
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        self.precision_score = precision_score(self.data["y_test"], self.y_pred_test)
        self.recall_score = recall_score(self.data["y_test"], self.y_pred_test)
        self.f1_score = f1_score(self.data["y_test"], self.y_pred_test)
        print(f'precision_score = {self.precision_score}')
        print(f'recall_score = {self.recall_score}')
        print(f'f1_score = {self.f1_score}')
    
    def test_data_class_proba(self):        
        false_proba = np.count_nonzero(self.data['y_test']) / self.data['y_test'].shape[0]
        true_proba = 1.0 - false_proba
        print(f'test set normal case probability: {false_proba}')
        print(f'test set alcohol case probability: {true_proba}')
        self.real_proba = dict()
        self.real_proba['false_proba'] = false_proba
        self.real_proba['true_proba'] = true_proba
    
    def predict_proba(self):
        self.y_proba_test = self.model.predict_proba(self.data['X_test'])[:,1]
        self.y_proba_train = self.model.predict_proba(self.data['X_train'])[:,1]

    def proba_thresholds(self):
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        # Vary thresholds by 0.05 from 0.05 to 0.95
        thresholds = np.arange(0.05, 1, 0.05)
        precisions = list()
        recalls = list()
        neg_recalls = list()
        for threshold in thresholds:
            # Apply threshold
            y_threshold = np.where(self.y_proba_test > threshold, 1, 0)
            # Calculate precision and recall
            precision = precision_score(self.data['y_test'], y_threshold)
            recall = recall_score(self.data['y_test'], y_threshold)
            neg_recall = recall_score(1-self.data['y_test'], 1-y_threshold)
            # Append to list
            precisions.append(precision)
            recalls.append(recall)
            neg_recalls.append(neg_recall)
        # get output
        self.proba_threshold = dict()
        self.proba_threshold['thresholds'] = thresholds
        self.proba_threshold['precisions'] = precisions
        self.proba_threshold['recalls'] = recalls
        self.proba_threshold['neg_recalls'] = neg_recalls
        # Visualize the result
        plt.figure()
        plt.plot(thresholds, precisions, label='precision', marker='o')
        plt.plot(thresholds, recalls, label='recall', marker='o')
        plt.xlim(0, 1)
        plt.xlabel('threshold')
        plt.ylabel('score')
        plt.legend()
        plt.show()
        
    def roc_auc(self):
        from sklearn.metrics import roc_curve, roc_auc_score
        # get roc auc train
        fprs_train, tprs_train, thresholds_train = roc_curve(self.data['y_train'], self.y_proba_train)
        roc_auc_train = roc_auc_score(self.data['y_train'], self.y_proba_train)
        # get roc auc test
        fprs_test, tprs_test, thresholds_test = roc_curve(self.data['y_test'], self.y_proba_test)
        roc_auc_test = roc_auc_score(self.data['y_test'], self.y_proba_test)
        # Plot the ROC curve.
        plt.figure()
        plt.plot(fprs_train, tprs_train, color='gray', lw=5, label='train', linestyle=' ', marker='.')
        plt.plot(fprs_test, tprs_test, lw=1, color='red', label='test')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='expected')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC and AUC')
        plt.legend(loc="best")
        plt.show()
        print(f"Test AUC score: {roc_auc_test}")
        print(f"Train AUC score: {roc_auc_train}")
        self.roc_auc_test = roc_auc_test
        self.roc_auc_train = roc_auc_train
    
    def standard(self):
        print('runing cross validation scores (this takes a while):')
        self.cross_val_score(5)
        print('\nfit model...')
        self.fitmodel()
        print('predict model...')
        self.predict()
        print('\naccuracy scores:')
        self.accuracy_score()
        print('\nconfusion matrix:')
        self.confusion_matrix()
        self.display_confusion_matrix()
        print('\nclassification report:')
        self.classification_report()
        self.print_evaluation_scores()
        print('\nreal data class proba:')
        self.test_data_class_proba()
        print('\n prediction proba')
        self.predict_proba()
        print('\n proba threshold analysis')
        self.proba_thresholds()
        print('\nROC AUC analysis')
        self.roc_auc()
    
    def save(self, savedir):
        # remove data from object to save space
        delattr(self, 'data')
        # get model name
        model_type = type(model_eval.model)
        model_type_str  = str(model_type)
        model_name_components = model_type_str.split('.')
        model_name = model_name_components[-1].replace("'>","")
        # save
        import pickle, os
        savepath = os.path.join(savedir, model_name+'_eval.pickle')
        pickle.dump(self, open(savepath, 'wb'))



