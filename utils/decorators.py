from functools import wraps
import pandas as pd
import os

from utils.constants import *

"""
    Making a decorator that accepts arguments was tricky.  Found a solution here: https://stackoverflow.com/questions/5929107/decorators-with-parameters
    
    A decorator to:
    * Append the target folder to the path
    * Confirm the existence of a file path
    * Avoid re-downloading if the file already exists and just return the dataframe instead

    Parameters:
    - folder: Based folder to check for output file prior to executing the method it wraps

    Wrapped File Parameters:
    - First argument must be the name of the output file.  This will be updated to have folder (above) prepended to the path
"""
def file_check_decorator(folder):
    def file_check(func):
        def wrapper(*args, **kwargs):
            filepath = folder + args[0]
            if os.path.exists(filepath):
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            try:
                # new set of args, using the filepath for the first one
                new_args = [filepath if idx == 0 else arg for idx, arg in enumerate(args)]
                result = func(*new_args, **kwargs)
            except Exception as ex:
                # clean up failed write
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise ex
            return result
        return wrapper
    return file_check