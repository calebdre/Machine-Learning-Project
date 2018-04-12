import numpy as np
import pandas as pd

from time import time

from sklearn.utils import shuffle
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from imblearn.over_sampling import SMOTE

from IPython.core.debugger import set_trace

def mprint(*args):
    for arg in args:
        print(arg)
        print(" ")

def merge(df, labels):
    return labels.merge(df, left_index=True,right_index=True)

class Experiment:
    def __init__(self, df, labels, model_class, model_config_names, model_config_init_values, k_folds=10):        
        self.data = df
        self.labels = labels
        self.label_values = labels["label"].unique()
        self.model_class = model_class
        self.model_config_names = model_config_names
        
        self.trial_num = 1
        self.init_experiments(model_config_names, model_config_init_values)
        
        self.k_folds = k_folds
    
    def tweak(self, parameter, new_value):
        self.experiments.loc[len(self.experiments) - 1][parameter] = new_value
        return self
    
    def run_trial(self):
        exp_num = self.experiments.shape[0]
        mprint("Running trail #{}\n------------------------------------".format(exp_num))

        performance_results = []
        folds, labels = self.split_k_folds()
        
        config = self.current_model_config()
        labels = pd.DataFrame(labels)
        
        for i, test_train in enumerate(folds):
            mprint(
                "*************************",
                "Running fold {} of {}".format(i+1, self.k_folds),
                "*************************"
            )
            
            test, train = test_train
            model = self.model()
            
            t1 = time()
            train = model.train(train, pd.DataFrame(labels), *config)
            mprint("fold {} took {}s".format(i+1, time() - t1))
            
            predictions = test.apply(lambda row: model.predict(row), axis=1)
            performance = self.measure_performance(predictions, merge(test, labels), self.label_values)
            
            self.record_trial(predictions, performance, train, test, i+1 == len(folds))
        
        self.trial_num += 1
        self.experiments.loc[len(self.experiments) - 1]["trial_num"] = self.trial_num
        return self.trial_results()
        
    def trial_results(self, trial_num=None):
        if trial_num == None:
            trial_num = self.experiments["trial_num"].max() - 1
        
        return self.experiments[self.experiments["trial_num"] == trial_num], self.experiment_data[int(trial_num)]    

#<--------  PRIVATE METHODS -------->
    def current_trial(self):
        return self.get_trial("current", "latest")
    
    def current_model_config(self):
        return self.experiments.loc[len(self.experiments) - 1][self.model_config_names].astype("int32").values
    
    def get_trial(self, trial_num=None, trial_instance=None):
        if trial_num == "current":
            if trial_instance == None:
                return self.experiments[self.experiments["trial_num"] == self.experiments["trial_num"].max()]
            elif trial_instance == "latest":
                latest = self.experiments[self.experiments["trial_num"] == self.experiments["trial_num"].max()]
                return latest.loc[len(latest) - 1]
        elif trial_num == "prev":
            if trial_instance != None:
                return self.experiment[self.experiments["trial_num"].max() - 1].iloc[trial_instance]
            else:
                return self.experiment[self.experiments["trial_num"].max() - 1]
        elif trial_num == None:
            return self.experiments[self.experiments["trial_num"] == self.experiments["trial_num"].max()]
        else:
            return self.experiments[self.experiments["trial_num"] == trial_num]
        
    def prev_trial(self):
        return self.get_trial("prev")
    
    def trial_data(self, trial_num=None):
        if trial_num == None:
            trial_num = self.experiments["trial_num"].max()-1
        
        return self.experiment_data[trial_num]
    
    def model(self):
        return self.model_class(verbose=False)
    
    def init_experiments(self, config_names, config_values):
        derived_cols = self.performance_measures() + ["trial_num"]
        all_cols = config_names + derived_cols
        first_row = config_values + [np.nan for i in derived_cols]
        
        self.experiments = pd.DataFrame(columns=all_cols)
        self.experiments.loc[0] = first_row
        self.experiments.loc[0]["trial_num"] = self.trial_num
        self.experiments[config_names] = self.experiments[config_names].fillna(-1)
        self.experiment_data = []
    
    def record_trial(self, predictions, perf_results, train, test, final=False):
        for key in perf_results:
            self.experiments.loc[len(self.experiments) - 1][key] = perf_results[key]

        self.experiments.loc[len(self.experiments)] = self.experiments.loc[len(self.experiments) - 1]
        self.experiment_data.append((train, test, predictions))
            
    def performance_measures(self):
        return ["log_loss", "class_accuracy", "precision", "recall", "f1-score", "support"]
    
    def split_k_folds(self):
        splitter = int(np.ceil(self.data.shape[0] / self.k_folds))
        df = shuffle(merge(self.data, self.labels))
        labels = df.pop("label")

        folds = []
        for i in range(1, self.k_folds+1):
            train = df.iloc[(i-1) * splitter: i * splitter]
            test = df.iloc[np.r_[0:(i-1) * splitter, i*splitter: df.shape[0]]]
            folds.append((train, test))

        return folds, labels
    
    def measure_performance(self, predictions, test_set, label_values=None):
        test_labels = test_set.pop("label")
        if label_values is None:
            label_values = test_labels.unique()

        precision, recall, fscore, support = score(test_labels, predictions, average='weighted')

        lvs = [[1 if p == 1 else 0 for l in label_values] for p in predictions]
        return {
            "log_loss" : log_loss(test_labels, lvs, normalize=True, labels=label_values),
            "class_accuracy": accuracy_score(test_labels, predictions, normalize=True),
            "precision": precision,
            "recall": recall,
            "f1-score": fscore,
            "support": support,
        }
        