import numpy as np

a= np.load('uniform.npy')
np.savetxt("foo.csv", a, delimiter=",")