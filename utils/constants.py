# Adjust to get less FRED data
DEFAULT_OBSERVATION_START = '1/1/1871'

# Pipeline data folders
RAW_DATA_PATH = './raw_data/'
CLEAN_DATA_PATH = './clean_data/'
TRAINING_DATA_PATH = './training_data/'

# Randomization
SEED = 42

# Possible training targets
CANDIDATE_TARGETS = ['bear', 'correction','Regime']
TARGET = 'bear'
REMOVE_LIST = [*CANDIDATE_TARGETS, 'Date']

# Training and evaluation
SCORING = "roc_auc"  # options: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
