import numpy as np
from scipy.io import mmwrite
from scipy.sparse import lil_matrix

prop_test = .1

if __name__ == '__main__':
  infile = open('subset.mtx', 'r')
  of_tst = open('subset_test.txt', 'w')

  for i in xrange(2):
    infile.readline()

  line = infile.readline().split()
  m, n, trn_n = [int(i) for i in line]

  train = lil_matrix((m, n))

  for line in infile:
    u = np.random.uniform()
    if u > prop_test:
      i, j, x = [int(i) for i in line.split()]
      train[i - 1, j - 1] = x
    else:
      of_tst.write(line)

  mmwrite('subset_train', train)
  infile.close()
  of_tst.close()
