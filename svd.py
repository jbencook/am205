import numpy as np
from scipy.io import mmread, mmwrite
import scipy.sparse.linalg as sla
import pandas as pd
import os
import time
from sys import argv
from subprocess import call

'''
Let's try some SVD!
'''

if __name__ == '__main__':
	data_file = argv[1]

	## The following is not robust! unpigz the compressed ratings matrix, 
	## read, and re-compress:
	#call(['unpigz', data_file])
	data = mmread(data_file).todense()
	#call(['pigz', data_file[:-3]])

	k = 1000
	u,s,v = np.linalg.svd(data[:k,:], full_matrices=True)

