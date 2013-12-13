import numpy as np
import pandas as pd
import os
from sys import argv

def get_user_ids(path, files, outfile):
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

if __name__ == '__main__':
	path       = argv[1]
	num_movies = argv[2]

	if num_movies == 'all':
		files = os.listdir(path)
	else:
		files = os.listdir(path)[:int(num_movies)]

	get_user_ids(path, files, 'user_ids.csv')

