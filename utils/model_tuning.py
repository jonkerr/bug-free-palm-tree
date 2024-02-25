import os

def get_fully_qualified_path(stage, feature, target, split_type):
    folder = f'./model_data/stage_{stage}/{target}_{split_type}'
    if feature is not None:
        folder += f'_{feature}'
    folder += '/'
    return folder


def get_simple_path(stage):
    return f'./model_data/stage_{stage}/'


def get_pickle_name(model_name, stage, feature, target, split_type, use_fully_qualified_path=True):
    '''
    The fully qualified path is interesting to save optimal model parameters for each stage/feature selection option/target/split type/etc.
    However, there is a concern that this could lead to overfitting.  Use the following test:
    1. Train all paramaters on optimal, per config setting
    2. Record outputs and find "top model"
    3. Run again with a different seed value.  The hyperparameters won't change since they've alread been saved but we can find out how sensitive the results are base on seed.
    4. If results are significantly different, then use the simple path to use the same settings for all configurations.
    '''
    folder = get_fully_qualified_path(stage, feature, target, split_type) if use_fully_qualified_path else get_simple_path(stage)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    return folder + model_name + '.pkl'

