import numpy as np
from util import *
from svmutil import *

x_train=read_csv('X_train.csv',5000)
y_train=read_csv('Y_train.csv',5000,True)
x_test=read_csv('X_test.csv',2500)
y_test=read_csv('Y_test.csv',2500,True)

params={'Linear: ':'0','Polynomial: ':'1','RBF':'2'}
acc=[]
for key,val in params.items():
    prob=svm_problem(y_train,x_train)
    param=svm_parameter('-t '+val)
    model=svm_train(prob,param)
    p_label,p_acc,p_val=svm_predict(y_test,x_test,model)
    acc.append(p_acc)

print('\nAccuracy')
i=0
for key in params:
    print(key,acc[i])
    i+=1
