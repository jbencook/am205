import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as sla
import pandas as pd
import os
import time
from sys import argv
from subprocess import call


if __name__ == '__main__':
	f = open(argv[1], 'r')
	m,n = int(argv[2]),int(argv[3])

	for i in xrange(3):
		print f.readline()

	data = lil_matrix((m,n), dtype=np.int8)

	for i,line in enumerate(f):
		if i % 1000 == 0:
			print i
		user,movie,rating = [int(i) for i in line.split()]

		if user <= m and movie <= n:
			data[user-1,movie-1] = rating

	print 'file read'
	mmwrite('subset', data)
