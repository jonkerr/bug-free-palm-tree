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
REMOVE = ['Date','bear','correction']

def get_data(split_date = '1980-01-01'):
    df = pd.read_csv(CLEAN_DATA_PATH + 'merged.csv', index_col=0, parse_dates=True).reset_index(names='Date')
    df_train, df_test = df[df['Date'] < split_date], df[df['Date'] >= split_date]
    
    X_train = df_train.drop(REMOVE, axis=1)
    y_train = df_train['bear']

    X_test = df_test.drop(REMOVE, axis=1)
    y_test = df_test['bear']
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

X_train, y_train, X_test, y_test = get_data()
