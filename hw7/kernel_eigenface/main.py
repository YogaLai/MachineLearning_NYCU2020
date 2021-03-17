import cupy as np
from util import *
import matplotlib.pyplot as plt
import sys

h=50
w=50
train_X,train_Y=load_dataset()
test_X,test_Y=load_dataset(train=False)
test_mu=np.mean(test_X,axis=1).reshape(-1,1)
n=len(train_Y)
kernel_type=sys.argv[1]

def show_face(eigenface):
    for i in range(25):
        img=eigenface[:,i].reshape(h,w)
        plt.subplot(5,5,i+1)
        plt.imshow(img,cmap='gray')
    plt.show()
        
def show_reconstruct(reconstruct):
    rand_idx=np.random.choice(n,10)
    for i in range(10):        
        plt.subplot(2,5,i+1)
        plt.imshow(reconstruct[:,rand_idx[i]].reshape(h,w),cmap='gray')
    plt.show()

def eigenface():
    print('Eigenface')
    eigen_vec,mu=pca(train_X)
    eigen_vec=eigen_vec[:,:25]
    show_face(eigen_vec)

    # reconstruct reference: learnopencv.com/face-reconstruction-using-eigenfaces-cpp-python/
    proj_data=eigen_vec.T@(train_X-mu)
    reconstruct=eigen_vec@proj_data+mu
    show_reconstruct(reconstruct)

    proj_test_data=eigen_vec.T@(test_X-test_mu)
    face_recognition(proj_data,proj_test_data,train_Y,test_Y)

def kernel_eigenface():
    print('Kernel Eigenface')
    X=np.hstack((train_X,test_X))
    proj_data=kerenl_pca(X,kernel_type)
    proj_train_data=proj_data[:,:len(train_Y)]
    proj_test_data=proj_data[:,len(train_Y):]
    face_recognition(proj_train_data,proj_test_data,train_Y,test_Y)

def fisherface():
    print('Fisherface')
    eigen_vec,mu=lda(train_X,train_Y)
    eigen_vec=eigen_vec[:,:25]
    show_face(eigen_vec)
    
    #reconstruction
    proj_train_data=eigen_vec.T@(train_X-mu)
    reconstruct=eigen_vec@proj_train_data+mu
    show_reconstruct(reconstruct)

    proj_test_data=eigen_vec.T@(test_X-test_mu)
    face_recognition(proj_train_data,proj_test_data,train_Y,test_Y)

def kernel_fisherface():
    print('Kernel Fisherface')
    X=np.hstack((train_X,test_X))
    Y=np.hstack((train_Y,test_Y))
    proj_data=kernel_lda(X,Y,kernel_type)
    proj_train_data=proj_data[:,:len(train_Y)]
    proj_test_data=proj_data[:,len(train_Y):]
    face_recognition(proj_train_data,proj_test_data,train_Y,test_Y)

# kernel_eigenface()
# kernel_fisherface()
eigenface()
fisherface()