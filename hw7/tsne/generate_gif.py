import os
import imageio
import sys
import cv2

method=sys.argv[1]
gif_buff=[]
for fn in os.listdir(method+'_imgs'):
    img=cv2.imread(method+'_imgs/'+fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gif_buff.append(img)
imageio.mimsave(method+'.gif',gif_buff,'GIF',duration=0.05)
