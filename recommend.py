from __future__ import division
##
import numpy as np
#from prettyplotlib import plt
from scipy.io import mmread
##
#from incremental_svd2 import incremental_SVD
#from svd_reconstruct import single_dot


def preprocess_recommender(m):
  # Remove sparsity by filling matrix with average movie rating
  # Normalize each entry by customer's average rating
  pass

if __name__ == '__main__':
  train = np.matrix(mmread('subset_train.mtx').todense())
  test = np.loadtxt('subset_test.txt')
  print 'Using matrix of size {}'.format(train.shape)
