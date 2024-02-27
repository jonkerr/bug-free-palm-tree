import pandas as pd

# constants
from utils.constants import (
    SPLIT_DATA_PATH, SPLIT_TYPE, FEATURE_TYPE, TARGET, SEED
)

# models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
)

baseline_models = [
    LogisticRegression(random_state=SEED),
    KNeighborsClassifier(),  # no random_state
    SVC(
        probability=True, random_state=SEED
    ),  # probability=True needed for roc_auc_score
    GaussianNB(),
    BernoulliNB(),
    GaussianProcessClassifier(random_state=SEED),
    DecisionTreeClassifier(random_state=SEED),
    RandomForestClassifier(random_state=SEED),
    GradientBoostingClassifier(random_state=SEED),
    AdaBoostClassifier(random_state=SEED),
    ExtraTreesClassifier(random_state=SEED),
    BaggingClassifier(random_state=SEED),
    xgb.XGBClassifier(random_state=SEED),
]

def get_training_data(split_type=SPLIT_TYPE, feature_type=FEATURE_TYPE, target=TARGET, verbose=True):
    """
    need to get files in the form:
    paths = ['X_train_std.csv', 'y_train_std.csv', 'X_test_std.csv', 'y_test_std.csv']
    """

    def format_name(fname):
        
        name = fname
        if feature_type is not None and not 'y_' in name:
            name += "_" + feature_type
            
        return f"{SPLIT_DATA_PATH}{name}_{target}_{split_type}.csv"
                
        #if feature_type:
        #    return f"{TRAINING_DATA_PATH}{fname}_{feature_type}_{split_type}.csv"
        #return f"{TRAINING_DATA_PATH}{fname}_{split_type}.csv"

    files = ["X_train", "y_train", "X_test", "y_test"]
    data = {fname: pd.read_csv(format_name(fname)) for fname in files}
    
    if verbose:
        print('\n----------------------------------------------------')
        if feature_type:
            print(f"**Data is using the '{feature_type}' feature type**")
        print(f"**Data is using the '{split_type}' split type**")
        print(f"**Data is using the '{target}' target**\n")
        
    return data
