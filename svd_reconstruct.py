from __future__ import division
##
import numpy as np
import scipy.linalg
from prettyplotlib import plt
#import matplotlib.pyplot as plt
from scipy.io import mmread
from sklearn.metrics import mean_squared_error
##
from incremental_svd2 import incremental_SVD


def single_dot(u, svT, x, y):
  colU = u[y, :]
  rowV = svT[:, x]
  return (colU).dot(rowV)


def check_orthogonality(A):
  return np.linalg.norm(A.T.dot(A) - np.diag([1] * min(A.shape)))

if __name__ == '__main__':
  train = np.matrix(mmread('subset_train.mtx').todense())
  train = train[0:2000, 0:100]
  print 'Using matrix of size {}'.format(train.shape)

  print 'Testing SVD'
  svdX = []
  svdY = []
  orthoX = []
  orthoY = []
  u, s, vT = scipy.linalg.svd(train)
  assert np.allclose(train, u.dot(scipy.linalg.diagsvd(s, u.shape[0], vT.shape[1]).dot(vT)))
  # See the loss in performance as we perform low-rank approximations
  for k in xrange(1, 100):
    low_s = [s[i] for i in xrange(k)] + (min(u.shape[0], vT.shape[1]) - k) * [0]
    reconstruct = u.dot(scipy.linalg.diagsvd(low_s, u.shape[0], vT.shape[1]).dot(vT))
    err = np.sqrt(mean_squared_error(train, reconstruct))
    print 'Exact SVD with low-rank approximation {}'.format(k)
    #print err
    #print
    svdX.append(k)
    svdY.append(err)
    orthoX.append(k)
    orthoY.append(check_orthogonality(u))
  plt.plot(svdX, svdY, label="SVD", color='black', linewidth='2', linestyle='--')

  print
  print 'Testing incremental SVD'
  incr_ortho = []
  for num in xrange(100, 1001, 300):
    print '... with block size of {}'.format(num)
    X, Y = [], []
    incr_orthoY = []
    for k in xrange(1, 101, 1):
      if k % 25 == 0:
        print '   ... up to k={}'.format(k)
      u, s, vT = incremental_SVD(train, k, num)
      reconstruct = u.dot(s.dot(vT))
      X.append(k)
      Y.append(np.sqrt(mean_squared_error(train, reconstruct)))
      incr_orthoY.append(check_orthogonality(u))
    incr_ortho.append(['iSVD u={}'.format(num), X, incr_orthoY])
    plt.plot(X, Y, label='iSVD u={}'.format(num))
  """
  print 'Testing raw SVD => exact reconstruction'
  svT = scipy.linalg.diagsvd(s, u.shape[0], vT.shape[1]).dot(vT)
  for y in xrange(train.shape[0]):
    for x in xrange(train.shape[1]):
      colU = u[y, :]
      rowV = svT[:, x]
      assert np.allclose(train[y, x], single_dot(u, svT, x, y))
  """
  ##
  plt.title('SVD reconstruction error on {}x{} matrix'.format(*train.shape))
  plt.xlabel('Low rank approximation (k)')
  plt.ylabel('Root Mean Squared Error')
  plt.ylim(0, max(svdY))
  plt.legend(loc='best')
  plt.savefig('reconstruct_error_{}x{}.pdf'.format(*train.shape))
  plt.show(block=True)
  ##
  plt.plot(svdX, svdY, label="SVD", color='black', linewidth='2', linestyle='--')
  for label, X, Y in incr_ortho:
    plt.plot(X, Y, label=label)
  plt.title('SVD orthogonality error on {}x{} matrix'.format(*train.shape))
  plt.xlabel('Low rank approximation (k)')
  plt.ylabel('Orthogonality error')
  plt.semilogy()
  #plt.ylim(0, max(orthoY))
  plt.legend(loc='best')
  plt.savefig('reconstruct_ortho_{}x{}.pdf'.format(*train.shape))
  plt.show(block=True)
