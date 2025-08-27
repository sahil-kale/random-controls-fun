import numpy as np

X = np.random.rand(5, 3) # random data matrix
A = np.array([[1, 0], [0.1, 2]])
U, S, V = np.linalg.svd(A, full_matrices=True) # full SVD
Uhat, Shat, Vhat = np.linalg.svd(A, full_matrices=False) # economy SVD

y = np.array([432, 543]) # find what effort moves into this direction
y_prime = U.T @ y # the vertical motion in U basis
x_prime = y_prime / S # find the required input to meet the output in U basis (y_prime)
x = V.T @ x_prime # convert the x_prime back to an input in original basis 
breakpoint()