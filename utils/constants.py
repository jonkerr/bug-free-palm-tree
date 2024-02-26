from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
# Adjust to get less FRED data
DEFAULT_OBSERVATION_START = '1/1/1871'
#POST_CLEANING_START_DATE = '1960-01-01'
POST_CLEANING_START_DATE = '1954-01-01'
#POST_CLEANING_START_DATE = '1939-01-01'

# Pipeline data folders
RAW_DATA_PATH = './raw_data/'
CLEAN_DATA_PATH = './clean_data/'
SPLIT_DATA_PATH = './split_data/'
TRAINING_DATA_PATH = './training_data/'
#FEATURE_DATA_PATH = './feature_data/'
MODEL_PERFORMANCE = './model_performance/'

# Randomization
SEED = 42

# Possible training targets
#CANDIDATE_TARGETS = ['bear', 'correction','Regime']
CANDIDATE_TARGETS = ['bear', 'Regime']

TARGET = 'bear'
REMOVE_LIST = [*CANDIDATE_TARGETS, 'Date']

# Training and evaluation
SCORING = "roc_auc"  # options: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
SPLIT_TYPE = "std"
SPLIT_TEST_SIZE = 0.25

#FEATURE_TYPE = "lasso"
FEATURE_TYPE = None #"lasso"

PERFORMANCE_METRICS =  {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
    }