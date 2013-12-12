#################################################################################
## This program cleans Netflix data and writes
## it to a CSV file.
## 
## Usage: python data-cleaning.py <path to movies> <number of movies to load>
## 
## Notes: - to load all the movies in the directory use 'all' for number
##          for number of movies. e.g.
##       
##          python data-cleaning.py download/training_set all
##			
##		  - you can find the Netflix data here:
##          
##          
#################################################################################

import numpy as np
import pandas as pd
import os
from sys import argv

#if __name__ == 'main':
path = argv[1]
num_movies = argv[2]

if num_movies == 'all':
	files = os.listdir(path)
else:
	files = os.listdir(path)[:int(num_movies)]

data = {}

ct = 0
for f in files:
	if ct % 100 == 0:
		print ct
	infile = open('%s/%s' % (path, f), 'r')
	movie = infile.readline().strip()[:-1]
	data[movie] = {}
	for line in infile:
		user,rating,_ = line.split(',')
		data[movie][user] = rating
	infile.close()
	ct += 1

outfile = 'data_%s.csv' % num_movies
pd.DataFrame(data).to_csv(outfile)


