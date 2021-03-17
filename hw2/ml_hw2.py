import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
import util
import MNIST
NONZEROVAR=100
NUM_CLASS=10
#https://github.com/chiha8888/NCTU-ML-class/blob/master/HW02/mode/continuous.py
def gaussian_prob(x,mu,var):
  return ((1/math.sqrt(2*math.pi*var))*math.exp(-(x-mu)**2/(2*var)))

def show_imagination_num(prob,nR,nC):
  print('Imagination of numbers in Bayesian classifier:')
  for c in range(NUM_CLASS):
    print(c+':')
    for i in range(nR*nC):
        if prob[c,i]>=128: 
          print('1',end=' ') 
        else: 
          print('0 ',end=' ')
        if i%28==0:
          print('')
    print('')

train_images,train_labels,nR,nC=util.read_MNIST_dataset('train-images.idx3-ubyte','train-labels.idx1-ubyte')
# train_labels=util.read_MNIST_label('train-labels.idx1-ubyte')
test_images,test_labels,nR,nC=util.read_MNIST_dataset('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')
# test_labels=util.read_MNIST_label('t10k-labels.idx1-ubyte')
(a,b),(c,d)=MNIST.load()
print(test_labels[-100:])
print(d[-100:])
# (train_images,train_labels),(test_images,test_labels)=MNIST.load()
# nR,nC=28,28
nImg=len(train_images)
# print(train_labels[0:10])

#Dicreate
prob=np.zeros((NUM_CLASS,nR*nC,32))
for i in range(nImg):
  c=train_labels[i]
  for j in range(nR*nC):
    prob[c][j][int(train_images[i,j]/8)]+=1

prior=[]
for c in range(NUM_CLASS):
  prior.append(len(train_images[train_labels==c])/nImg)
  for j in range(nR*nC):
    total=sum(prob[c][j])
    prob[c][j]/=total

error=0
posterior=np.zeros(NUM_CLASS)
for i in range(len(test_labels)):
# for i in range(10):
  print('Posterior ( in log scale ):')
  for c in range(NUM_CLASS):
    likelihood=0
    for j in range(nR*nC):
      prob_val=prob[c,j,int(test_images[i,j]/8)]
      if prob_val!=0:
        likelihood+=math.log(prob_val)
        # likelihood+=math.log(max(1e-4,prob_val))
    posterior[c]=likelihood+math.log(prior[c])
  for c in range(NUM_CLASS):
      print('%d: %f'%(c,posterior[c]/sum(posterior)))
  print('Prediction: %d, Ans: %d\n'%(np.argmax(posterior),test_labels[i]))
  if np.argmax(posterior)!=test_labels[i]:
    error+=1
  
  # show_imagination_num(posterior,nR,nC)
print('Error rate: ',error/len(test_labels))

'''
# Calculate Gaussian
gaussian=np.zeros((10,nR*nC,256))  
prior=[]
for c in range(NUM_CLASS):
  imgs=train_images[train_labels==c]
  prior.append(len(imgs)/nImg)
  for i in range(nR*nC):
    mu=np.mean(imgs[:,i])
    var=np.var(imgs[:,i])
    if var==0:
      var=NONZEROVAR
    for j in range(256):
      gaussian[c,i,j]=gaussian_prob(j,mu,var)

#Calculate posterior
posterior=np.zeros(NUM_CLASS)
for i in range(3):
  print('Posterior ( in log scale ):')
  for c in range(NUM_CLASS):
    likelihood=0
    for j in range(nR*nC):
      gaussian_val=gaussian[c,j,test_images[i,j]]
      if gaussian_val!=0:
        likelihood+=math.log(gaussian_val)
    posterior[c]=likelihood+math.log(prior[c])
  posterior/=sum(posterior)
  for c in range(NUM_CLASS):
    print('%d: %f'%(c,posterior[c]))
  print('Prediction: %d, Ans: %d\n'%(np.argmin(posterior),test_labels[i]))
  '''