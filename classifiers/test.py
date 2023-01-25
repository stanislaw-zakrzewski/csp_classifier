import numpy as np
from sklearn.decomposition import TruncatedSVD

# Create a matrix of data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Initialize the TruncatedSVD model
svd = TruncatedSVD(n_components=2)

# Fit the model to the data
svd.fit(data)

# Print the factors
print("U:", svd.components_)
print("S:", svd.singular_values_)