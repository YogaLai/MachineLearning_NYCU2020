import csv
import numpy as np 

def read_csv(path,len,label=False):
    with open(path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      if not label:
        arr=np.zeros((len,784))
      else:
        arr=np.zeros(len)
      for i,row in enumerate(rows):
        arr[i]=np.asarray(row)
    return arr