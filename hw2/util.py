import struct as st
import numpy as np

def read_MNIST_dataset(img_path,label_path):
    img_file=open(img_path,'rb')
    label_file=open(label_path,'rb')
    img_file.read(8)
    # nImg=int.from_bytes(img_file.read(4),byteorder='big')
    nR=int.from_bytes(img_file.read(4),byteorder='big')
    nC=int.from_bytes(img_file.read(4),byteorder='big')

    label_file.read(4)
    nLabel=int.from_bytes(label_file.read(4),byteorder='big')
    imgs=np.zeros((nLabel,nR*nC),dtype='uint8')
    labels=np.zeros(nLabel,dtype='uint8')
    for i in range(nLabel):
        labels[i]=int.from_bytes(label_file.read(1),byteorder='big')
        for j in range(nR*nC):
            imgs[i,j]=int.from_bytes(img_file.read(1),byteorder='big')

    img_file.close()
    label_file.close()
    return imgs,labels,nR,nC

def read_MNIST_label(filename):
    f=open(filename,'rb')
    f.read(4)
    n=int.from_bytes(f.read(4),byteorder='big')
    labels=np.zeros(60000,dtype='uint8')
    for i in range(n):
        labels[i]=int.from_bytes(f.read(1),byteorder='big')
    f.close()
    return labels