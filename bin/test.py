import numpy as np 

tab = []

tab.append((20,30))
tab.append((15,18))
tab = np.array(tab)


I = np.array([[1,2,3], [1,14,19], [20, 16, 18]])
J = np.array([[1,8,3], [1,40,18], [20, 17, 18]])


K = np.reshape(I[:,0],(3,1))
print(K + J)
# X = np.array([1,4,0,8])
# a = np.argmin(X)
# X[a] = 10000 
# print(X)
# b = np.argmin(X)
print (J.shape)