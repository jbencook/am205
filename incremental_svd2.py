import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread, mmwrite
from sklearn.metrics import mean_squared_error
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

def test_perf(X, test, uk, sk, vk):
    r = np.zeros(X.shape[0], dtype=np.float32)
    
    for i in xrange(len(r)):
        r[i] = np.mean(X[i,np.argwhere(X[i,:] != 0)])
    
    prediction = np.zeros(test.shape[0])
    for n in xrange(test.shape[0]):
        i,j = test[i,:2].astype(np.int32)
        prediction[n] = r[i] + np.dot(np.dot(uk,np.sqrt(sk).T)[i,:], np.dot(np.sqrt(sk),vk)[:,j])
        
    return prediction

if __name__ == '__main__':
	train = np.asarray(mmread('subset_train.mtx').todense())
	test  = np.loadtxt('subset_test.txt')

	u,s,v = incremental_SVD(train, 6, 100, by_row=True)

	pred = test_perf(train, test, u, s, v)
	print np.sqrt(mean_squared_error(pred,test[:,2]))


