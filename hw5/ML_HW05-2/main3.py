import numpy as np
from util import *
from svmutil import *
from scipy.spatial.distance import cdist

def kernel(Xn,Xm,gamma=2e-3):
    linear=Xn@Xm.T
    RBF=np.exp(-gamma*cdist(Xn,Xm,'sqeuclidean'))
    precomputed_kernel=linear+RBF
    precomputed_kernel=np.hstack((np.arange(1,len(Xn)+1).reshape(-1,1),precomputed_kernel))
    return precomputed_kernel

def kernel2(Xn,Xm,gamma=2e-3):
    c=np.random.rand()
    d=np.random.rand()
    polynomial=(Xn@Xm.T+c)**d
    RBF=np.exp(-gamma*cdist(Xn,Xm,'sqeuclidean'))
    precomputed_kernel=polynomial+RBF
    precomputed_kernel=np.hstack((np.arange(1,len(Xn)+1).reshape(-1,1),precomputed_kernel))
    return precomputed_kernel

x_train=read_csv('X_train.csv',5000)
y_train=read_csv('Y_train.csv',5000,True)
x_test=read_csv('X_test.csv',2500)
y_test=read_csv('Y_test.csv',2500,True)


# precomputed_kernel=kernel(x_train,x_train)
precomputed_kernel=kernel2(x_train,x_train)
print(precomputed_kernel.shape)
prob=svm_problem(y_train,precomputed_kernel,isKernel=True)
param=svm_parameter('-t 4')
model=svm_train(prob,param)

precomputed_kernel=kernel(x_test,x_train)
p_label,p_acc,p_val=svm_predict(y_test,precomputed_kernel,model)
print('\n\n',p_acc)