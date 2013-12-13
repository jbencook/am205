import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread, mmwrite
import scipy.sparse.linalg as sla
import pandas as pd
import os
import time
from sys import argv
from subprocess import call

def incremental_SVD(X, k, num, by_row=True):
	if by_row:
		u,s,v = np.linalg.svd(X[:num,:])
		uk = u[:,:k]
		sk = np.diag(s[:k])
		vk = v[:k,:]
		sk_inv = np.linalg.pinv(sk)

		for i in xrange(X.shape[0] - num):
			c = X[num + i,:]
			cp = np.dot(np.dot(c,vk.T), sk_inv)
			uk = np.vstack((uk, cp))

	else:
		u,s,v = np.linalg.svd(X[:,:num])
		uk = u[:,:k]
		sk = np.diag(s[:k])
		vk = v[:k,:]
		sk_inv = np.linalg.pinv(sk)

		for j in xrange(X.shape[1] - num):
			p  = X[:,num + j]
			pp = np.dot(np.dot(p.T,uk), sk_inv)
			vkk = np.zeros((vk.shape[0],vk.shape[1] + 1))
			vkk[:,:vk.shape[1]] = vk
			vkk[:,vk.shape[1]] = pp
			vk = vkk

	return uk, sk, vk		

if __name__ == '__main__':
	data = np.asarray(mmread('subset.mtx').todense())

	data = data[:2000,:500]

	u,s,v = incremental_SVD(data, 6, 100, by_row=False)

	print u.shape
	print s.shape
	print v.shape