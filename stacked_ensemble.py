
import os
import pandas as pd
import pickle
from abc import ABC, abstractmethod

# sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# local
from utils.model_tuning import get_pickle_name
from utils.constants import PERFORMANCE_METRICS, SPLIT_TEST_SIZE, SEED
from utils.training import get_training_data, baseline_models

import warnings
class StackedEnsembleBase(BaseEstimator, ABC):
    '''
    StackedEnsembleBase is an sklearn compatible estimator that provides the base functionality for various subclasses.
    The expecation is that subclasses will define criteria for their own stage1/sttage2 models.

    Abstract methods that subclass needs to implement:

    _initialize_stage1_models()
        - Select the list of models used and the hyperparameter tunings for each

    _initialize_stage2_model()
        - Configure the best performing stage 2 model with the preconfigured optimal hyperparameters for the selected dataset        
    '''

    def __init__(self, verbose=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.model_needs_scaling = [
            "LogisticRegression",
            "KNeighborsClassifier",
            "SVC",
            "GaussianProcessClassifier",
        ]
#        self.ignore_in_phase_2 = [
#            'DecisionTreeClassifier',
#            'AdaBoostClassifier',
#        ]
        self.stage1_models = self.__initialize_stage1_models__()
        self.stage2_model = self.__initialize_stage2_model__()
        self.scalar = StandardScaler()

        
    def print_debug(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
        
        
    @abstractmethod
    def get_feature_set():
        pass
    
    def __get_pickled_model__(self, model_name, stage, feature_set):
            pkl_name = get_pickle_name(model_name, stage, feature_set['feature'], feature_set['target'], feature_set['split_type'])
            if os.path.exists(pkl_name):
                with open(pkl_name, "rb") as file:
                    return pickle.load(file)


    def __initialize_stage1_models__(self) -> list:        
        fs = self.get_feature_set()        
        tuned_models = {}
        for model in baseline_models:
            model_name = model.__class__.__name__      
#            if model_name in self.ignore_in_phase_2:
#                continue                        
#            tuned_models.append(self.__get_pickled_model__(model_name, 1, fs)) 
            tuned_models[model_name] = self.__get_pickled_model__(model_name, 1, fs)
            
        return tuned_models
    

    def __initialize_stage2_model__(self):
        '''
        Returns: the best performing stage 2 model with the preconfigured optimal hyperparameters for the selected dataset        
        '''
        fs = self.get_feature_set()
        return self.__get_pickled_model__(fs['s2_model'], 2, fs)  
    
    
    def remove_noisey_models(self, X, y):
        '''
        Sometimes we end up with models with really bad f1 scores.  
        Even though we have an extremely limited number of records, let's see if we can correctly 
        identify which modes are behaving badly and remove them from given their input to the stage 2 model.
        --
        Params
         - X - really this is X_train but we're going to split so we have a test/validate set
         - y - really this is y_train but we're going to split so we have a test/validate set
        '''
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)

        # get a fresh set of models from pickle
        models = self.__initialize_stage1_models__()
        noisey_models = []
        
        for model_name, model in models.items():
            self.print_debug("Testing: ", model_name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = f1_score(y_test, preds)
            if score==0:
                self.print_debug("Dropping: ", model_name)
                noisey_models.append(model_name)
            
        # drop noisey models
        [self.stage1_models.pop(key) for key in noisey_models]
    

    def fit(self, X_train, y_train):
        '''
        Train the multi-model based on X_train, y_train.
        For sub-models that reqire scaling, also train the scaler
        '''
        X_train_scaled = self.scalar.fit_transform(X_train)
        probs = pd.DataFrame()

#        print(y_train)
#        return

        # find and remove models that have an f1_score == 0 - e.g. useless estimators
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.remove_noisey_models(X_train, y_train)
            
        # train stage 1 models
        self.print_debug('Fitting Stage 1')
        for model in self.stage1_models.values():
            model_name = model.__class__.__name__            
            # use scaled data if model needs it
            training_data = X_train_scaled if model_name in self.model_needs_scaling else X_train
            model.fit(training_data, y_train)            
            probs[model_name] = model.predict_proba(training_data)[:, 1]

        # fit stage 2 model
        probs = probs.round(4)
        self.print_debug('Fitting Stage 2')
        self.stage2_model.fit(probs, y_train)

    def fit_transform(self, X_train, y_train):
        '''
        Included for BaseEstimator compliance.  It's really just a wrapper for fit, followed by transform.
        '''
        self.fit(X_train, y_train)
        return self.predict(X_train)
    

    def _predict(self, X, use_probs):
        '''
        Whether we're predicting probs or binary outcomes, only the last step is different while the rest is the same.
        As such, this will be implemented once, with a boolean value to determine which outcome is desired.        
        '''
        X_scaled = self.scalar.transform(X)
        probs = pd.DataFrame()

        # test stage1 models
        self.print_debug('Predicting Stage 1')
        for model in self.stage1_models.values():
            model_name = model.__class__.__name__
            
            # use scaled data if model needs it
            X_data = X_scaled if model_name in self.model_needs_scaling else X
            probs[model_name] = model.predict_proba(X_data)[:, 1]

        self.print_debug('Predicting Stage 2')
        probs = probs.round(4)
        if use_probs:
            return self.stage2_model.predict_proba(probs)
        else:
            return self.stage2_model.predict(probs)

    def predict(self, X):
        '''
        Predict a binary label based on supplied values of X
        '''
        return self._predict(X, use_probs=False)

    def predict_proba(self, X):
        '''
        Predict probability of label based on supplied values of X
        '''
        return self._predict(X, use_probs=True)


class BearStackedEnsemble(StackedEnsembleBase):
    
    def __init__(self,verbose=True) -> None:
        super().__init__(verbose)
        
        
    def get_feature_set(self):
        return {
            'feature': None, 
            'target': 'bear', 
            'split_type': 'std',    
            's2_model': 'BaggingClassifier'        
        }
        


class RegimeStackedEnsemble(StackedEnsembleBase):

    def __init__(self,verbose=True) -> None:
        super().__init__(verbose)

    def get_feature_set(self):
        return {
            'feature': None, 
            'target': 'Regime', 
            'split_type': 'std',    
            's2_model': 'AdaBoostClassifier'        
        }

   
def test_model(model: StackedEnsembleBase):
    '''
    Adapted from select_models.py
    '''
    fs = model.get_feature_set()
       
    # get data
    data = get_training_data(fs['split_type'], fs['feature'], fs['target'])
    X_train, y_train = data["X_train"], data["y_train"]            
    X_test, y_test = data["X_test"], data["y_test"]
    
    # fit
    model.fit(X_train, y_train.values.ravel())  
    
    # get metrics
    results = {}    
    Y_pred = model.predict(X_test)  # Test the model
    Y_pred_proba = model.predict_proba(X_test)
    
    for metric_name, metric_func in PERFORMANCE_METRICS.items():
        if metric_name == "roc_auc":
            score = metric_func(y_test, Y_pred_proba)
        else:
            score = metric_func(y_test, Y_pred)
        results[metric_name] = round(score, 4)
        
        #print(f'{metric_name}: {results[metric_name]}')
    
    df = pd.DataFrame(results, index=[0])
    print(df)  

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-t',
        '--target',
        help='Which split target? [bear|rec|all] Default is all',
        default="all",
        required=False
    )

    args = parser.parse_args()
    
    verbose = False
    if args.target in ['bear','all']:
        test_model(BearStackedEnsemble(verbose))
    if args.target in ['rec','all']:
        test_model(RegimeStackedEnsemble(verbose))
