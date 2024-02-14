# Make process based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

all: predict

# Targets for removing data
removeraw:
	rm -fr raw_data

removecleaned: removefeatures removemodeldata
	rm -fr clean_data

removemodeldata:
	rm -fr training_data 
#	rm -fr model_data

removefeatures:
	rm -fr feature_data

removeall: removemodeldata removefeatures removecleaned removeraw	

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
reclean: removecleaned clean
remodel: removemodeldata selectmodels
remodelc: removemodeldata removecleaned selectmodels
