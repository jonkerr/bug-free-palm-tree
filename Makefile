# Make process based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

all: get_data #clean_data

clean:
# remove old data

get_data:
	python get_data.py


#clean_data:
# python clean_data.py

cleanraw:
	rm -f raw_data/recession.csv
	rm -f raw_data/dataset.csv


# remove all data
cleanall: cleanraw clean_data
