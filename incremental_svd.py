import numpy as np
from scipy.io import mmread
from sklearn.metric import mean_squared_error


def cf_recommend(X, k, num_rows=6, test_size=.1):
    '''
    This function performs incremental SVD on X.
    Note:

    k is the number of singular values
    num_rows is the number of rows on which
    to do full SVD
    test_size is the proportion of ratings
    to hold out as a test set
    '''

    IJ = np.argwhere(X != 0)
    holdout = np.random.randint(0, len(IJ), np.round(len(IJ) * test_size))

    testset = np.zeros((len(holdout), 4))

    for ct, ndx in enumerate(holdout):
        i, j = IJ[ndx]
        testset[ct, :-1] = i, j, X[i, j]
        X[i, j] = 0.

    u, s, v = np.linalg.svd(X[:num_rows, :])

    uk = u[:, :k]
    sk = np.diag(s[:k])
    vk = v[:k, :]
    sinv = np.linalg.pinv(sk)

    for i in xrange(X.shape[0] - num_rows):
        c = X[num_rows + i, :]
        cp = np.dot(np.dot(c, vk.T), sinv)
        uk = np.vstack((uk, cp))

    return uk, sk, vk, testset


def test_rec(X, testset, uk, sk, vk):
    '''
    test the collaborative filtering recommendation system
    '''
    r = np.zeros(X.shape[0], dtype=np.float32)

    for i in xrange(len(r)):
        r[i] = np.mean(X[i, np.argwhere(X[i, :] != 0)])
    for n in xrange(testset.shape[0]):
        i, j = testset[i, :2].astype(np.int32)
        testset[n, 3] = r[i] + np.dot(np.dot(uk, np.sqrt(sk).T)[i, :], np.dot(np.sqrt(sk), vk)[:, j])

    return testset

if __name__ == '__main__':
    #Usage:
    data = mmread('subset.mtx').todense()

    uk, sk, vk, testset = cf_recommend(data, 10, 2000)
    testset = test_rec(data, testset, uk, sk, vk)

    ndces = ~np.isnan(testset[:, 3])
    np.sqrt(mean_squared_error(testset[ndces, 2], testset[ndces, 3]))
