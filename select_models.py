'''
Select models and optimal hyper-parameters to be used in ensemble model.
----
Lots used from: EDA_Spike/Part 4_Model_v1.ipynb
'''
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

CLEAN_DATA_PATH = './clean_data/'
TRAINING_PATH = './training_data/'

def get_training_data(split_type='std'):
    '''
    need to get files in the form:
    paths = ['X_train_std.csv', 'y_train_std.csv', 'X_test_std.csv', 'y_test_std.csv']
    '''
    def format_name(fname):
        return f'{TRAINING_PATH}/{fname}_{split_type}.csv'
    
    files = ['X_train', 'y_train', 'X_test', 'y_test']
    data = {fname: pd.read_csv(format_name(fname)) for fname in files}
    return data
    

data = get_training_data()
print(data['X_train'].shape)
