from PIL import Image
import os
import cv2
import numpy as np
import sys


def cut_allbbox():
    BasePath = 'Shelf/videos'   #路径
    campath_ = os.listdir(BasePath)  #读文件夹
    campath_.sort(key=lambda x: int(x.replace("camera","")))  #按数字排序
    # print(campath_)
    VPath = 'Shelf/videos/camera00'
    vpath_ = os.listdir(VPath)
    ALL = np.load("code/yolooutput.npz")  #读文件
    ALL = ALL['arr_0']  #[[mainframe, max], bbox]
    img_ = []
    img_gallery = []
    img_query = []
    lenth = len(vpath_)  #总帧数
    
    for index in range(lenth):  #第index帧图像
        im = []  #存某一帧下所有视角的图像
        imbbox = ALL[index][1]  #存某一帧下所有视角所有人bbox
        
        for i in range(5):   #第i个视角
            path_img = os.listdir(os.path.join(BasePath, campath_[i]))
            path_img.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  #第i个视角下所有图像排序
            ImgPath = cv2.imread(os.path.join(BasePath, campath_[i], path_img[index]))
            # print(path_img[index])
            im.append(path_img[index])  #某一帧下所有视角
        print(f"第{index}帧图像信息添加完成")
        
        mainframe, Num = ALL[index][0]   #人数最多视角 和 query人数
        bbox_main = ALL[index][1][mainframe][0]  #人数最多视角的第一个人
        if bbox_main:  
            querymain = (im[mainframe], 0, mainframe, 0)
        img_query.append([])  #第index帧
        img_gallery.append([])
        img_.append([])
        n = 0
        for i in range(Num):  #添加query  #视角
            query = (im[mainframe], i, mainframe, n)
            n = n + 1
            img_query[-1].append(query)
        n = 0
        for j in range(5):  #添加gallery  #视角
            imframe = Image.open(os.path.join(BasePath, campath_[j], path_img[index]))
            t = len(ALL[index][1][j])  #index帧第j个视角的bbox个数
            for k in range(t):
                gallery = (im[j], k, j, n)
                n = n + 1
                img_gallery[-1].append(gallery)
                sin_bbox = imbbox[j][k]
                cut_img = imframe.crop((float(sin_bbox[0]), float(sin_bbox[1]), float(sin_bbox[2]), float(sin_bbox[3])))
                resize_img = cut_img.resize((64,128), Image.ANTIALIAS).convert('RGB')
                img_[-1].append(resize_img) 
        
    # print(img_gallery)
    # print(img_query)
    return img_, img_gallery, img_query
                  
def cut_singlebbox(index, camlist):
    BasePath = 'Shelf/videos'   #路径
    campath_ = os.listdir(BasePath)  #读文件夹
    campath_.sort(key=lambda x: int(x.replace("camera","")))  #按数字排序
    ALL = np.load("code/yolooutput.npz")  #读文件
    ALL = ALL['arr_0']  #[[mainframe, max], bbox]
    img_ = []
    img_gallery = []
    img_query = []
    
    im = []  #存某一帧下所有视角的图像
    imbbox = ALL[index][1]  #存某一帧下所有视角所有人bbox
    
    for i in camlist:   #第i个视角
        path_img = os.listdir(os.path.join(BasePath, campath_[i]))
        path_img.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  #第i个视角下所有图像排序
        ImgPath = cv2.imread(os.path.join(BasePath, campath_[i], path_img[index]))
        # print(path_img[index])
        im.append(path_img[index])  #某一帧下所有视角
    print(f"第{index}帧图像信息添加完成")
    
    mainframe, Num = ALL[index][0]   #人数最多视角 和 query人数
    
    nn = 0
    path_img = os.listdir(os.path.join(BasePath, campath_[mainframe]))
    path_img.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))
    Frame = Image.open(os.path.join(BasePath, campath_[mainframe], path_img[index]))
    
    if mainframe in camlist:
        for ii in range(Num):  #添加query  #视角
            bbox = imbbox[mainframe][ii]
            cut_ = Frame.crop((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
            resize_ = cut_.resize((64,128), Image.ANTIALIAS).convert('RGB')
            img_.append(resize_)
            query = (f"query_{nn}", ii, mainframe, nn)
            img_query.append(query)
            nn = nn + 1
    else: 
        find_main = []
        for i in camlist:
            find_main.append(len(ALL[index][1][i]))
        mainframe = find_main.index(max(find_main))
        for ii in range(len(ALL[index][1][mainframe])):  #添加query  #视角
            bbox = imbbox[mainframe][ii]
            cut_ = Frame.crop((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
            resize_ = cut_.resize((64,128), Image.ANTIALIAS).convert('RGB')
            img_.append(resize_)
            query = (f"query_{nn}", ii, mainframe, nn)
            img_query.append(query)
            nn = nn + 1
    
    for j in camlist:  #添加gallery  #第j个视角
        if j != mainframe:
            path_img = os.listdir(os.path.join(BasePath, campath_[j]))
            path_img.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))
            imframe = Image.open(os.path.join(BasePath, campath_[j], path_img[index])) #帧
            t = len(imbbox[j])  #index帧第j个视角的bbox个数
            for k in range(t):  #第k个人
                sin_bbox = imbbox[j][k]
                cut_img = imframe.crop((float(sin_bbox[0]), float(sin_bbox[1]), float(sin_bbox[2]), float(sin_bbox[3])))
                resize_img = cut_img.resize((64,128), Image.ANTIALIAS).convert('RGB')
                img_.append(resize_img)
                gallery = (f"gallery_{nn-Num}", k, j, nn)
                img_gallery.append(gallery)
                nn = nn + 1
                
    return img_, img_gallery, img_query, mainframe


def Gallery_index(index, Q_num):
    BasePath = 'Shelf/videos'   #路径
    campath_ = os.listdir(BasePath)  #读文件夹
    campath_.sort(key=lambda x: int(x.replace("camera","")))  #按数字排序
    path_img = os.listdir(os.path.join(BasePath, campath_[0]))
    path_img.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))
    ALL = np.load("code/yolooutput.npz")  #读文件
    ALL = ALL['arr_0']  #[[mainframe, max], bbox]
    imbbox = ALL[index][1]  #存某一帧下所有视角所有人bbox
    
    nn = Q_num
    Gallery = []
    G_img = []
    for j in range(5):  #第j个视角
        imframe = Image.open(os.path.join(BasePath, campath_[j], path_img[index])) #帧
        t = len(imbbox[j])  #index帧第j个视角的bbox个数
        for k in range(t):  #第k个人
            bbox = imbbox[j][k]
            cut_img = imframe.crop((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
            resize_img = cut_img.resize((64,128), Image.ANTIALIAS).convert('RGB')
            G_img.append(resize_img)
            gallery = (f"gallery_{nn-Q_num}", k, j, nn)
            Gallery.append(gallery)
            nn += 1
            
    return Gallery, G_img, nn
    
    
def Gallery_index_(index, camlist):
    BasePath = 'Shelf/videos'   #路径
    campath_ = os.listdir(BasePath)  #读文件夹
    campath_.sort(key=lambda x: int(x.replace("camera","")))  #按数字排序
    path_img = os.listdir(os.path.join(BasePath, campath_[0]))
    path_img.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))
    ALL = np.load("code/yolooutput.npz")  #读文件
    ALL = ALL['arr_0']  #[[mainframe, max], bbox]
    imbbox = ALL[index][1]  #存某一帧下所有视角所有人bbox
    
    nn = 0
    Gallery = []
    G_img = []
    for j in camlist:  #第j个视角
        imframe = Image.open(os.path.join(BasePath, campath_[j], path_img[index])) #帧
        t = len(imbbox[j])  #index帧第j个视角的bbox个数
        for k in range(t):  #第k个人
            bbox = imbbox[j][k]
            cut_img = imframe.crop((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
            resize_img = cut_img.resize((64,128), Image.ANTIALIAS).convert('RGB')
            G_img.append(resize_img)
            gallery = (f"gallery_{nn}", k, j, nn)
            Gallery.append(gallery)
            nn += 1
            
    return Gallery, G_img, nn
