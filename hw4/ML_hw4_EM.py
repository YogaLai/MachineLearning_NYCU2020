import struct as st
import numpy as np
import math
#參考網站: http://blog.manfredas.com/expectation-maximization-tutorial/

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

trainset,labels,nR,nC=read_MNIST_dataset('train-images.idx3-ubyte','train-labels.idx1-ubyte')

#Binning the gray level value into two bins
trainset=np.asarray(trainset>=128,dtype='uint8')


def E_step(X,pi,p):
  w=np.zeros((len(X),10))
  for i in range(len(X)):
    for c in range(num_classes):
        # w[i,c]=np.sum(X[i]*p[c]+(1-X[i])*(1-p[c]))
        w[i,c]=np.prod(X[i]*p[c]+(1-X[i])*(1-p[c]))
  w=w*pi.reshape(1,-1)
  # w=w+pi.reshape(1,-1)
  sums = np.sum(w,axis=1).reshape(-1,1)
  sums[sums==0] = 1
  w = w/sums
  return w

# def E_step(X,pi,P):

def M_step(X,w):
  w_sum = np.sum(w, axis=0)
  w_sum[w_sum==0]=1
  new_pi = w_sum / w.shape[0] 
  # new_p = (np.matmul(X.T, w) / w_sum).T
  new_p=(X.T@(w/w_sum)).T
  return new_pi, new_p

def print_num(arr):
  for i in range(nR):
      for j in range(nC):
          print('%.0f'%arr[i,j],end=' ')
      print()
  print()
  print()

def print_predict_num(p,pi):
  pixel=np.zeros(nR*nC)
  for c in range(num_classes):
    print('Class %d:'%c)
    for i in range(nR*nC):
      pixel[i]=1 if p[c,i]>=0.5 else 0
    print_num(pixel.reshape(nR,nC))


def print_labelel_num():
  for c in range(num_classes):
    idx=np.argwhere(labels==c)
    output=np.zeros(nR*nC)
    for i in range(nR*nC):
      output[i]=np.sum(trainset[idx,i])/len(idx)
    output=output.reshape(nR,nC)
    output=np.round(output)
    print('labeled class %d:'%c)
    print_num(output)

n=nR*nC
num_classes=10
p = np.random.rand(num_classes, n)
pi=np.ones(num_classes)/num_classes
max_iter=250
iter=0
converge=1e-7

# last_diff,diff=1000,100
# eps=1
# while abs(last_diff-diff)>eps and diff>eps:
while True:
  iter+=1
  print('Calculate E step...')
  w=E_step(trainset,pi,p)
  print('Calculate M step...')
  new_pi,new_p=M_step(trainset,w)
  delta=sum(sum(abs(new_p-p)))+sum(abs(new_pi-pi))
  # last_diff=diff
  # diff=np.sum(np.abs(new_pi-pi))+np.sum(np.abs(new_p-p))
  print_predict_num(new_p,new_pi)
  print('No. of Iteration: %d, Difference: %f'%(iter,delta))
  print()
  print('-----------------------------------------------------')
  print()
  if iter>=500:
    print("Greater than max_iter")
    break
  elif delta<converge:
    print('Coverge')
    break

  # elif np.alltrue(abs(new_pi-pi)<=converge) and np.alltrue(abs(new_p-p)<=converge):
  #   print('Coverage')
  #   break
  pi=new_pi
  p=new_p

print_labelel_num()

def confusion_matrix(predict):
  tp_cnt=0
  for i in range(num_classes):
    TP=np.count_nonzero(labels[predict==i]==i)
    FP=np.count_nonzero(labels[predict==i]!=i)
    TN=np.count_nonzero(labels[predict!=i]!=i)
    FN=np.count_nonzero(labels[predict!=i]==i)

    print('Confusion matrix %d'%i)
    print('\t\t\tPredict number %d Predict not number %d'%(i,i))
    print('Is number %d\t\t%d\t\t\t%d'%(i,TP,FN))
    print('Isn\'t number %d\t\t%d\t\t\t%d\n'%(i,FP,TN))
    print('Sensitivity (Successfully predict number 0): %f'%(TP/(TP+FN)))
    print('Specificity (Successfully predict not number 0): %f\n'%(TN/(TN+FP)))
    print('---------------------------------')

    tp_cnt+=TP
  return tp_cnt

predict=np.argmax(w,axis=1)
tp_cnt=confusion_matrix(predict)
print('Total iteration to converage: %d'%iter)
errors=1-tp_cnt/60000
print('Total error rate: %f'%(errors))
