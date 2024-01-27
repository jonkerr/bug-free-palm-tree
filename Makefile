# Make process based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

all: getdata cleandata predict


cleandata: 
	python clean_data.py

clean:
# remove old data

getdata:
	python get_data.py


cleanraw:
	rm -f raw_data/recession.csv
	rm -f raw_data/dataset.csv

# remove all data
cleanall: cleanraw 

selectmodels:
	python select_models.py


predict: selectmodels
	python predict.py