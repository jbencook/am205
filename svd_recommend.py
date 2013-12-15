from __future__ import division
##
import numpy as np
import scipy.linalg
from prettyplotlib import plt
#import matplotlib.pyplot as plt
from scipy.io import mmread
from sklearn.metrics import mean_squared_error
##
from incremental_svd2 import incremental_SVD, test_perf


def test_perf(X, test, uk, sk, vk, r):
  prediction = np.zeros(test.shape[0])
  for n in xrange(test.shape[0]):
    i, j = test[n, :2] - 1
    prediction[n] = r[i] + np.dot(np.dot(uk, np.sqrt(sk).T)[i, :], np.dot(np.sqrt(sk), vk)[:, j])

  return prediction


def get_error(U, S, V, train, test, r):
  #U, S, V = incremental_SVD(train, k, u, by_row=True)
  pred = test_perf(train, test, U, S, V, r)
  ndx = ~np.isnan(pred)
  rmse = np.sqrt(mean_squared_error(pred[ndx], test[ndx, 2]))
  #ortho = np.linalg.norm(U.dot(U.T) - np.identity(U.shape[0]))
  return rmse

if __name__ == '__main__':
  raw_train = np.matrix(mmread('subset_train.mtx').todense(), dtype=np.float)
  test = np.loadtxt('subset_test.txt', dtype=np.int32)
  maxX, maxY = 2000, 100
  train = raw_train[0:maxX, 0:maxY]
  test = test[0:maxX, 0:maxY]
  print 'Using matrix of size {}'.format(train.shape)

  #
  prod_avg = np.zeros(train.shape[0], dtype=np.float32)
  for i in xrange(len(prod_avg)):
    prod_avg[i] = np.mean(train[i, np.argwhere(train[i, :] != 0)])
  #
  cust_avg = np.zeros(train.shape[1], dtype=np.float32)
  for i in xrange(len(cust_avg)):
    cust_avg[i] = np.mean(train[np.argwhere(train[:, i] != 0), i])
  # Replace empties by average product rating
  for i, avg in enumerate(prod_avg):
    train[i, np.argwhere(train[i, :] == 0)] += avg
  # Normalize all ratings by the averge customer rating
  for i, avg in enumerate(cust_avg):
    train[:, i] -= avg
  print 'Using matrix of size {}'.format(train.shape)

  print train[0:10, 0:10]

  """
  print 'Testing SVD'
  svdX = []
  svdY = []
  u, s, vT = scipy.linalg.svd(train)
  for k in xrange(1, 100):
    low_s = [s[i] for i in xrange(k)]  # + (min(u.shape[0], vT.shape[1]) - k) * [0]
    print 'Exact SVD with low-rank approximation {}'.format(k)
    svdX.append(k)
    svdY.append(get_error(u, np.diag(low_s), vT, train, test))
  plt.plot(svdX, svdY, label="SVD", color='black', linewidth='2', linestyle='--')
  """

  print
  print 'Testing incremental SVD'
  for num in xrange(400, 1001, 300):
    print '... with block size of {}'.format(num)
    X, Y = [], []
    for k in xrange(1, 91, 10):
      print k
      u, s, vT = incremental_SVD(train, k, num)
      X.append(k)
      Y.append(get_error(u, s, vT, train, test, prod_avg))
    plt.plot(X, Y, label='iSVD u={}'.format(num))
  ##
  plt.title('Recommendation system RMSE on {}x{} matrix'.format(*train.shape))
  plt.xlabel('Low rank approximation (k)')
  plt.ylabel('Root Mean Squared Error')
  #plt.ylim(0, max(svdY))
  plt.legend(loc='best')
  plt.savefig('recommend_rmse_{}x{}.pdf'.format(*train.shape))
  plt.show(block=True)
