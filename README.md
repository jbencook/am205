am205
=====

Incremental SVD

## Data cleaning and generation

The entire ratings matrix is being hosted in a sparse and compressed format on my Bluehost server. It can be downloaded with the following command:

wget http://jbenjamincook.com/jbc/ratings_matrix.mtx.gz

This can be decompressed with the following:

unpigz ratings_matrix.mtx.gz

Or if pigz is not installed on your system, you can use:

gunzip ratings_matrix.mtx.gz

## Utils training and test set

To get a subset of the dataset, we run:

python utils/subset-data.py (path to ratings_matrix.mtx) (m) (n)

e.g.

python utils/subset-data.py ~/ratings_matrix.mtx 3000 1000

To split it into a training and test set, you can use:

python utils/train_test_split.py

## Raw SVD, reconstruction error

python svd_reconstruct.py

## Mean Absolute Error and deviations from orthogonality

python incremental_svd2.py