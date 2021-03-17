import math

def factorial(n):
    fact=1
    for i in range(1,n+1):
        fact=fact*i
    return fact


a=int(input("Please input a: "))
b=int(input("Please input b: "))
f=open('online_learning_data.txt','r')
idx=1
for line in f:
    m=0
    line=line.replace('\n','')
    n=len(line)
    for item in line:
        if item=='1':
            m+=1
    p=m/n
    print('case %d: %s'%(idx,line))
    print('Likelihood: ',(p**m)*((1-p)**(n-m))*factorial(n)/(factorial(n-m)*factorial(m)))
    print('Beta prior: a=%d b=%d'%(a,b))
    print('Beta posterior: a=%d b=%d'%(a+m,b+n-m))
    print('\n')
    idx+=1
    a+=m
    b+=n-m

f.close()