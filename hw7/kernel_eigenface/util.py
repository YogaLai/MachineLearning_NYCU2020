import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import operator
from scipy.spatial.distance import cdist

def load_dataset(train=True):
    if train:
        root='Yale_Face_Database/Training/'
    else:
        root='Yale_Face_Database/Testing/'
    X=[]
    Y=[]
    dir=os.listdir(root)    
    for fn in dir:
        img=Image.open(root+fn)
        img=np.asarray(img.resize((50,50))).flatten()
        X.append(img)
        Y.append(int(fn[7:9]))
    X=np.asarray(X).T
    Y=np.asarray(Y)
    return X,Y

def pca(X):
    mu=np.mean(X,axis=1).reshape(-1,1)
    A=X-mu
    eigen_val, eigen_vec = np.linalg.eig(A.T@A)
        
    idx = np.argsort(-eigen_val)
    eigen_vec = eigen_vec[:,idx].real
    eigen_vec=A@eigen_vec
    eigen_vec_norm=np.linalg.norm(eigen_vec,axis=0)
    eigen_vec=eigen_vec/eigen_vec_norm

    return eigen_vec,mu

def kerenl_pca(X,kernel_type):
    kernel=get_kernel(X,kernel_type)
    N=kernel.shape[0]
    one_n=np.full((N,N),1/N)
    kernel=kernel-one_n@kernel-kernel@one_n+one_n@kernel@one_n
    eigen_val, eigen_vec = np.linalg.eigh(kernel)
    idx = np.argsort(-eigen_val)
    eigen_vec = eigen_vec[:,idx][:,:25].real
    eigen_vec_norm=np.linalg.norm(eigen_vec,axis=0)
    eigen_vec=eigen_vec/eigen_vec_norm

    return eigen_vec.T@kernel

def lda(X,Y):
    n=X.shape[0]
    num_subject=15
    mu=np.mean(X,axis=1).reshape(-1,1)
    Sw=np.zeros((n,n),dtype=np.float64)
    Sb=np.zeros((n,n),dtype=np.float64)

    for i in range(num_subject):
        X_i=X[:,Y==i+1]
        class_mean=np.mean(X_i,axis=1).reshape(-1,1)
        Sw+=(X_i-class_mean)@(X_i-class_mean).T
        Sb+=X_i.shape[1]*((mu-class_mean)@(mu-class_mean).T)
    
    S=np.linalg.pinv(Sw)@Sb
    eigen_val,eigen_vec=np.linalg.eig(S)

    idx = np.argsort(-eigen_val)
    eigen_vec = eigen_vec[:,idx]
    eigen_vec=np.asarray(eigen_vec.real)
    eigen_vec_norm=np.linalg.norm(eigen_vec,axis=0)
    eigen_vec=eigen_vec/eigen_vec_norm

    return eigen_vec,mu

def kernel_lda(X,Y,kernel_type):
    num_subject=15
    kernel=get_kernel(X,kernel_type)
    mu=np.mean(kernel,axis=1).reshape(-1,1)
    n=kernel.shape[0]
    Sw=np.zeros((n,n),dtype=np.float64)
    Sb=np.zeros((n,n),dtype=np.float64)
    
    for i in range(num_subject):
        k_i=kernel[:,Y==i+1]
        class_mean=k_i.mean(axis=1).reshape(-1,1)
        l=np.count_nonzero(Y==i+1)
        one_l=np.full((l,l),1/l)
        Sw+=k_i@(np.identity(l)-one_l)@k_i.T
        Sb+=l*(class_mean-mu)@(class_mean-mu).T

    S=np.linalg.pinv(Sw)@Sb
    eigen_val,eigen_vec=np.linalg.eig(S)

    eigen_vec_norm=np.linalg.norm(eigen_vec,axis=0)
    eigen_vec=eigen_vec/eigen_vec_norm
    idx = np.argsort(-eigen_val)
    eigen_vec = eigen_vec[:,idx][:,:25].real
    eigen_vec=np.asarray(eigen_vec.real)
    return eigen_vec.T@kernel

def face_recognition(proj_train_data,proj_test_data,train_Y,test_Y):
    #knn
    correct=0
    k=3
    distance=np.zeros(len(train_Y))
    for i in range(len(test_Y)):
        for j in range(len(train_Y)):
            distance[j]=np.sum(np.square(proj_test_data[:,i]-proj_train_data[:,j]))
        idx=np.argsort(distance)
        idx=idx[:k]
        label=np.unique(test_Y)
        cnt_dict=dict((y,0) for y in label)
        for l in range(k):
            cnt_dict[train_Y[idx[l]]]+=1
        predict=max(cnt_dict.items(), key=operator.itemgetter(1))[0]
        if predict==test_Y[i]:
            correct+=1
    
    print('The accuracy rate(%d/%d): %f' % (correct,len(test_Y),correct/len(test_Y)*100))

def rbf_kernel(X,sigma=1e-7):
    return np.exp(-sigma*cdist(X.T, X.T, 'sqeuclidean'))

def poly_kernel(X,coef,degree):
    return np.power((X.T @ X) + coef, degree)

def get_kernel(X,kernel_type):
    if kernel_type=='rbf':
        kernel=rbf_kernel(X)
    elif kernel_type=='poly':
        kernel=poly_kernel(X,10,3)
    return kernel
    
