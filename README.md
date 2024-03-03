# Ensemble Model Prediction of Economic Downturns
MADS Milestone 2 Project

## Getting Started
### Preconditions
* [Python](https://www.python.org/downloads/) is installed.
* [Conda](https://docs.conda.io/projects/miniconda/en/latest/index.html) is installed.
* For Windows, [GnuWin32](https://sourceforge.net/projects/getgnuwin32/) is installed and path is configured.

### Configuring Conda

To configure your conda environment with all the same libraries as compatible with this project, run:
```
conda env create -f environment.yml
```

As with all conda environments, be sure to activate the (newly created) conda environment before using:
```
conda activate mads_milestone2
```

If using VS Code, be sure to use the same for Jupyter kernel

```
Select kernel -> Python Environments -> mads_milestone2
```

### Getting a FRED key

To access FRED data, you must [request an API key](https://fredaccount.stlouisfed.org/login/secure/).  Once you have the key:

1. Copy/rename hidden-dist.py to hidden.py
2. Put your API key information in the place that's listed.


More infor on FRED:
* [API documentation](https://fred.stlouisfed.org/docs/api/fred/)  
* [Tutorial](https://mortada.net/python-api-for-fred.html)



## General workflow

### Building Code
Makefile is used to simplify building, deleting and rebuilding data and models. 

To build the project end to end, simply run the make command with no targets:
```
make
```

### Iterating on models or data
Most likely, we're going to be proceeding in an iterative fashion that requires us to regularly blow away and recreate data.

For starting fresh and re-downloading all data (useful to do prior to submitting to ensure we haven't missing anything):
**WARNING: The following target can potentially be very slow as it will re-download everything from scratch.**
```
make refresh
```

For the data cleaning step, use:
```
make reclean
```

For model tuning:
```
make remodel
```


### Contributing to Git
Upon starting a new development session, it's a best practice to start with whatever the latest code is.  You can do this by running the following in a terminal
```
git pull
```

#### Ready to commit
Prior to checking in, you'll want to grab the latest in case someone else was developing at the same time.
```
git stash
git pull
git stash pop
```

Now you can commit select the files you want to stage and commit. (Recommend using VS Code for this step.)  

Finally, always remember to push your changes back to the repo:
```
git push
```