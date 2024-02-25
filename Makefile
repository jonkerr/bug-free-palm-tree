# Make process based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

all: predict

# Targets for removing data
removeraw:
	rm -fr raw_data

removecleaned: removesplit
	rm -fr clean_data

removesplit: removefeatures
	rm -fr split_data 

removefeatures:
	rm -fr feature_data

removeperf:
	rm -fr model_performance

removeall: removeperf removefeatures removesplit removecleaned removeraw	

# Targets for building things
getdata:
	python get_data.py

clean: getdata
	python clean_data.py

split: clean
	python split_data.py

features: split
	python select_features.py

selectmodels: features
	python select_models.py

predict: selectmodels
	python predict.py

# Composite targets for iterative development
reraw: removeraw getdata
refresh: removeall all
reclean: removecleaned clean
resplit: removesplit split
remodel: removefeatures selectmodels
remodelc: removecleaned selectmodels

# clean and then all from different places
recleana: removecleaned all
resplita: removesplit all

# lasso tuning
tunelasso:
	python select_features.py --feature_option lasso
	python select_models.py --feature_option lasso --split_target all

tunelasso2b:
	python select_features.py --feature_option lasso
	python select_models.py --feature_option lasso --split_target bear --run_type stg2

tunelasso2r:
	python select_features.py --feature_option lasso
	python select_models.py --feature_option lasso --split_target rec --run_type stg2

modelall: features
	python select_models.py --feature_option all --split_target all --run_type stg2
