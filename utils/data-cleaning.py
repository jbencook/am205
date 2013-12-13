import numpy as np
from scipy.io import mmwrite
from scipy.sparse import lil_matrix
import pandas as pd
import os
from sys import argv

'''
This program cleans Netflix data and writes it to a CSV file.

Usage: python data-cleaning.py <path to movies> <number of movies to load>
'''

def write_user_ids(path, files, outfile):
	users = set()
	i = 0
	of = open(outfile, 'w')
	for ndx,f in enumerate(files):
		if ndx % 100 == 0:
			print ndx
		infile = open('%s/%s' % (path, f), 'r')
		infile.readline().strip()[:-1]
		for line in infile:
			id, _, _ = line.split(',')
			if id not in users:
				users.add(id)
				of.write('%d,%s\n' % (i, id))
				i += 1
	print '%d total users' % (i + 1)
	of.close()

def get_user_dict(fname):
	f = open(fname, 'r')
	user_dict = {}
	for line in f:
		i,id = line.strip().split(',')
		user_dict[id] = int(i)
	f.close()
	return user_dict

if __name__ == '__main__':
	path       = argv[1]
	num_movies = argv[2]
	user_file  = argv[3]
	#user_file = 'user_ids.csv'

	if num_movies == 'all':
		files = os.listdir(path)
	else:
		files = os.listdir(path)[:int(num_movies)]

	write_user_ids(path, files, user_file)
	users = get_user_dict(user_file)
	data  = lil_matrix((len(users), len(files)))

	ct = 0
	for f in files:
		if ct % 100 == 0:
			print ct
		infile = open('%s/%s' % (path, f), 'r')
		j = int(infile.readline().strip()[:-1]) - 1 #Move number
		for line in infile:
			id,rating,_ = line.split(',')
			i = users[id]
			data[i,j] = int(rating)
		infile.close()
		ct += 1

	mmwrite('ratings_matrix', data)