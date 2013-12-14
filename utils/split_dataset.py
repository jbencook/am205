from __future__ import division
##
import random
import sys

# Enable repeatability
random.seed(42)

if __name__ == '__main__':
  train = open('subset_train.mtx', 'w')
  test = open('subset_test.mtx', 'w')

  for i, line in enumerate(sys.stdin):
    # Copy the same headers to both files
    if i < 3:
      train.write(line)
      test.write(line)
      continue
    # Otherwise, split lines between train and test
    if random.random() < 0.9:
      train.write(line)
    else:
      test.write(line)
