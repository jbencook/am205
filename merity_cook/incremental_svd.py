import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from sklearn.metrics import mean_squared_error


def incremental_SVD(X, K, num, by_row=True):
  '''
  Takes a vector of k values!
  '''
  if by_row:
    u, s, v = np.linalg.svd(X[:num, :])

    uk_list = []
    sk_list = []
    vk_list = []
    for k in K:
      uk = u[:, :k]
      sk = np.diag(s[:k])
      vk = v[:k, :]
      sk_inv = np.linalg.pinv(sk)

      for i in xrange(X.shape[0] - num):
        c = X[num + i, :]
        cp = np.dot(np.dot(c, vk.T), sk_inv)
        uk = np.vstack((uk, cp))

      uk_list.append(uk)
      sk_list.append(sk)
      vk_list.append(vk)

  return uk_list, sk_list, vk_list


def test_perf(X, test, uk, sk, vk):
    r = np.zeros(X.shape[0], dtype=np.float32)

    for i in xrange(len(r)):
        r[i] = np.mean(X[i, np.argwhere(X[i, :] != 0)])

    prediction = np.zeros(test.shape[0])
    for n in xrange(test.shape[0]):
        i, j = test[n, :2] - 1
        prediction[n] = r[i] + np.dot(np.dot(uk, np.sqrt(sk).T)[i, :], np.dot(np.sqrt(sk), vk)[:, j])

    return prediction


def get_error(K, u, train, test):
  RMSE = []
  ORTHO = []
  U, S, V = incremental_SVD(train, K, u, by_row=True)
  for i, k in enumerate(K):
    pred = test_perf(train, test, U[i], S[i], V[i])
    ndx = ~np.isnan(pred)
    print 'k = %d' % k
    RMSE.append(np.sqrt(mean_squared_error(pred[ndx], test[ndx, 2])))
    ORTHO.append(np.linalg.norm(U[i].T.dot(U[i]) - np.diag([1] * min(U[i].shape))))

  return RMSE, ORTHO

if __name__ == '__main__':
  train = np.asarray(mmread('subset_train.mtx').todense())
  test  = np.loadtxt('subset_test.txt', dtype=np.int32)

  num = 0
  for i in xrange(train.shape[0]):
    if np.all(train[i, :] == 0):
      num += 1
  print 'all zeros', num

  K = range(3, 50)
  nums = [100, 500, 1000, 2000, 3000]

  RMSE = []
  ORTHO = []

  for u in nums:
    print 'u = %d' % u
    rmse, ortho = get_error(K, u, train, test)
    RMSE.append(rmse)
    ORTHO.append(ortho)

  fig = plt.figure()
  for rmse in RMSE:
    plt.plot(K, rmse)
  plt.xlabel('low-rank approximation (k)')
  plt.ylabel('root mean squared error')
  leg = plt.legend(nums)
  leg.draw_frame(False)
  fig.savefig('RMSE.png')

  fig = plt.figure()
  for ortho in ORTHO:
    plt.plot(K, ortho)
  plt.xlabel('low-rank approximation (k)')
  plt.ylabel('deviation from orthogonality')
  leg = plt.legend(nums)
  leg.draw_frame(False)
  fig.savefig('ORTHO.png')
