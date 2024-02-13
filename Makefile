# Make process based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

all: predict

# Targets for removing data
removeraw:
	rd /s /q raw_data

removecleaned:
	rd /s /q clean_data

removemodeldata:
	rd /s /q training_data
	rd /s /q model_data

removeall: removemodeldata removecleaned removeraw

# Targets for building things
getdata:
	python get_data.py

clean: getdata
	python clean_data.py

features: clean
	python select_features.py

split: features
	python split_data.py

selectmodels: split
	python select_models.py

predict: selectmodels
	python predict.py

# Composite targets for iterative development
reraw: removeraw getdata
refresh: removeall all
reclean: removecleaned modeldata
remodel: removemodeldata selectmodels
remodelc: removemodeldata removecleaned selectmodels
