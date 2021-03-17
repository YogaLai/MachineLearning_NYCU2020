import numpy as np
import argparse
import cv2
from kernel import kernel
from numba import jit,NumbaWarning
import warnings
from plot import make_gif
import imageio
from scipy.spatial.distance import cdist
import random

warnings.simplefilter('ignore', category=NumbaWarning)
parser = argparse.ArgumentParser()
parser.add_argument('-k', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('-m')
args = parser.parse_args()

img_name=args.name
method=args.m
img=cv2.imread(img_name+'.png')

n_cluster=int(args.k)
n_points=10000
colormap=[[255,0,0],[0,0,255],[0,255,0]]
img=img.reshape(n_points,3)
cluster=np.zeros(n_points,dtype=np.int)
K=kernel(img)

def init(method='random'):
    if method=='random':
        for i in range(n_points):
            cluster[i]=random.randint(0,n_cluster-1)
    elif method=='kmeans++':        
        # 第一個群中心為隨機，計算每個點到與其最近的中心點的距離為dist
        # 以正比於dist的概率，隨機選擇一個點作為中心點加入中心點集中，重復直到選定k個中心點
        mean=np.zeros((n_cluster,2),dtype=np.int)
        mean[0]=[random.randint(0,n_points-1)/100,random.randint(0,n_points)%100]
        for k in range(1,n_cluster):
            max_distance=-1
            max_x_idx=0
            max_y_idx=0
            for i in range(n_points):
                x=i/100
                y=i%100
                distance=(mean[k-1,0]-x)**2+(mean[k-1,1]-y)**2
                if distance>max_distance:
                    max_distance=distance
                    max_x_idx=x
                    max_y_idx=y
            mean[k]=[max_x_idx,max_y_idx]

        #依照與群中心的距離來初始化每個data point所屬的class
        for i in range(n_points):
            distance=np.zeros(n_cluster)
            for j in range(n_cluster):
                x=i//100
                y=i%100
                distance[j]=(mean[j,0]-x)**2+(mean[j,1]-y)**2
            cluster[i]=np.argmin(distance)

    else:
        print('Unknow method')

def second_term(img_idx,cluster_idx):
    cluster_size=0
    result=0
    for i in range(n_points):
        if cluster[i]==cluster_idx:
            cluster_size+=1
            result+=K[img_idx,i]

    if cluster_size!=0:
        result=-2/cluster_size*result
    return result

@jit
def third_term():
    result=np.zeros(n_cluster)
    for k in range(n_cluster):
        for p in range(n_points):
            for q in range(n_points):
                if cluster[p]==k and cluster[q]==k:
                    # S=[p,q]
                    # C=[img[p],img[q]]
                    result[k]=result[k]+K[p,q]
    for k in range(n_cluster):
        cluster_size=np.count_nonzero(cluster==k)
        if cluster_size!=0:
            result[k]=result[k]/(cluster_size**2)
    return result

@jit
def kernel_kmeans():
    third=third_term()
    distance=np.zeros(n_cluster)
    print('calculate kernel kmaens ...')
    for i in range(n_points):
        for j in range(n_cluster):
            distance[j]=second_term(i,j)+third[j]
        min_idx=np.argmin(distance)
        cluster[i]=min_idx

def draw():
    color=np.zeros((n_points,3),dtype=np.uint8)
    for i in range(n_points):
        color[i]=colormap[cluster[i]]
    color=color.reshape(100,100,3)  
    return color


init(method)
delta=10
gif_buff=[]
iter=0
while delta>5:
    iter+=1
    print('Iter: ',iter)
    pre_cluster=cluster.copy()
    kernel_kmeans()
    color=draw()
    gif_buff.append(color)
    delta=np.sum(abs(cluster-pre_cluster))
    print('Delta: ',delta)
    print('----------------------------------------')

imageio.mimsave('kernel_kmeans_'+args.k+'_'+method+'_'+img_name+'.gif',gif_buff,'GIF',duration=0.1)
