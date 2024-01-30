# Make process based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

all: getdata modeldata predict

# Targets for removing data
removeraw:
	rm -f raw_data/recession.csv
	rm -f raw_data/dataset.csv

removecleaned:
	rm -fr clean_data

removemodeldata:
	rm -fr model_data

removeall: removemodeldata removecleaned removeraw

# Targets for building things
getdata:
	python get_data.py

modeldata: getdata
	python clean_data.py

selectmodels: modeldata
	python select_models.py

predict: selectmodels
	python predict.py

# Composite targets for iterative development
refresh: removeall all
reclean: removecleaned modeldata
remodel: removemodeldata selectmodels
