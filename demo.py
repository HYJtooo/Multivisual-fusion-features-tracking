from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

BasePath = 'Shelf/videos'  
campath_ = os.listdir(BasePath)  
campath_.sort(key=lambda x: int(x.replace("camera","")))  
path_img = os.listdir(os.path.join(BasePath, campath_[0])) 
path_img.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  
outbox = np.load("result/outBbox2.npz")  
outbox = outbox['arr_0']

bgr = [(255,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(0,69,255),(255,127,36)]
ALLbox_ = []
c0box = []
c1box = []
c2box = []
c3box = []
c4box = []

for i in range(len(outbox)):
    Draw = []
    for c in range(5): 
        pic_ = cv2.imread(os.path.join(BasePath, campath_[c], path_img[i]))
        cv2.putText(pic_, text=f"index_{i}_cam_C{c}", org=(10,30), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=bgr[5], thickness=2)
        for j in outbox[i][c]:
            ALLbox_.append(j[0])
            if c == 0:
                c0box.append(j[0])
            elif c == 1:
                c1box.append(j[0])
            elif c == 2:
                c2box.append(j[0])
            elif c == 3:
                c3box.append(j[0])
            else: 
                c4box.append(j[0])
            bbox = j[0]
            cv2.rectangle(pic_, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), bgr[j[1]], thickness=2)
            cv2.putText(pic_, f"{j[1]}", org=((int((int(bbox[2])+int(bbox[0]))/2))-10, int(bbox[1])+20), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255,255,255), thickness=2)
        Draw.append(pic_)
    Draw.append(np.ones((776, 1032, 3),np.uint8)*255)
    cv2.putText(Draw[5], text=f"index = {i}", org=(30,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=bgr[5], thickness=2)
    
    show1 = Draw[0]
    show2 = Draw[3]
    for n, d in enumerate(Draw):
        if 0 < n < 3:
            show1 = np.concatenate((show1, d),axis=1)
        elif 3 < n < 6:
            show2 = np.concatenate((show2, d),axis=1)
    showView = np.concatenate((show1,show2),axis=0)
    showView = cv2.resize(showView,(int(1032*1.5), 776))
    
    cv2.imshow("show", showView)
    cv2.waitKey(1)
            
