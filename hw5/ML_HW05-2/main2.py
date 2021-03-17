import numpy as np
from util import *
from svmutil import *

x_train=read_csv('X_train.csv',5000)
y_train=read_csv('Y_train.csv',5000,True)
x_test=read_csv('X_test.csv',2500)
y_test=read_csv('Y_test.csv',2500,True)

C=gamma=['0.5','0.3','0.1','1','3','5']
best_acc=0
for c in C:
    for g in gamma:
        prob=svm_problem(y_train,x_train)
        param=svm_parameter('-v 3 -t 2 -c '+c+' -g '+g)
        acc=svm_train(prob,param)
        print(acc)
        if acc>best_acc:
            best_acc=acc
            optimal_c=c
            optimal_g=g

print('The parameter C is: %s\nThe parameter gamma is: %s\nThe accuracy is: %f' % (optimal_c,optimal_g,best_acc))


