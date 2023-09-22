from matplotlib.pyplot import table
import numpy as np
import cv2
import argparse
import pickle as pkl
import torch
from torchvision.transforms import Normalize
import torchgeometry as tgm
from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants
import sys
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, type=bool,  help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,  help='Use cpu nms')
args = parser.parse_args()
box_censcl = np.load('result/shelf_censcl_out_7.npz')
box_censcl = box_censcl['arr_0']
bgr = [(238,0,0),(0,127,225),(50,205,50),(144,41,205),(255,0,0),(0,69,255),(255,127,36)]
camlist = [0,1,2,3,4]
fmt_calib = "Shelf/calib/P{camera:}.txt"
CAMERA_MATRIXS_TO_IMG = np.array([np.loadtxt(fmt_calib.format(camera=camera), delimiter=",") for camera in camlist])
ourpoints = np.load("result/all_3Dpots.npz")
ourpoints = ourpoints['arr_0']
smpl_connection = [ [0,1],[0,2],[0,3],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[3,6],[6,9],[9,13],
                    [9,14],[13,16],[16,18],[18,20],[20,22],[14,17],[17,19],[19,21],[21,23],
                    [9,12],[12,15]  ]
O_Joints = [15,16,20,14,17,21,13,18,22,12,19,23,1,10,11,0,5,6,4,7,3,8,2,9]


def process_image(img_file, center,scale, input_res=224):
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() 
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img   

if __name__ =='__main__':
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load trained model
    p_net = hmr(config.SMPL_MEAN_PARAMS)
    p_net.to(device)
    checkpoint = torch.load(args.trained_model)
    p_net.load_state_dict(checkpoint['model'], strict=False)
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)
    p_net.eval()
    # Generate rendered image
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES,faces=smpl.faces) 
    
    Bathpath = 'Shelf/videos'
    campath_ = os.listdir(Bathpath) 
    campath_.sort(key=lambda x: int(x.replace("camera","")))  
    img_path = os.listdir(os.path.join(Bathpath, campath_[0])) 
    img_path.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  
    VPath = 'Shelf/videos/camera00'
    vpath_ = os.listdir(VPath)
    lenth = len(vpath_) 
    
    modelpath = 'log_Shelf/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    with open(modelpath, 'rb') as file:
        model = pkl.load(file, encoding='iso-8859-1')
    J_re = model['J_regressor_prior'].todense()
    J_re_arr = np.array(J_re)
    last_po_3d = []
    nwrong = []  
    app_ = []  
    thread = 0.3
    fig = plt.figure()
    ax = Axes3D(fig)
    allps_3Dpots = []
    
    for ind in range(300):
        Draw5 = []
        Draw5.append(np.ones((20, 3150, 3),np.uint8)*255)
        Dr_1 = []
        Dr_1.append(np.ones((1570, 92, 3),np.uint8)*255)
        Dr_3 = []
        Dr_11 = []
        Dr_11.append(np.ones((65, 1920, 3),np.uint8)*255)
        indpoints = []
        cpoints = []
        for ap in range(len(app_)):  
            app_[ap] = 0
        path = f"out3D_{ind}.png"
        pic3D = cv2.imread(path)
        
        # for ops, opsinf in enumerate(ourpoints[ind]):
        #     ou_id = opsinf[0]
        #     bgr_ = (bgr[ou_id][2]/255, bgr[ou_id][1]/255, bgr[ou_id][0]/255)
        #     for opt, opots in enumerate(opsinf[1]):
        #         if isinstance(opots,list):
        #             oupoint = opots
        #         else: 
        #             oupoint = opots.tolist()
        #         if opt in O_Joints:
        #             ax.scatter(oupoint[0], oupoint[1], oupoint[2], c=bgr_, marker='o')
        #     for jo in smpl_connection:
        #         po0 = opsinf[1][jo[0]]
        #         po1 = opsinf[1][jo[1]]
        #         ax.plot([po0[0],po1[0]],[po0[1],po1[1]],[po0[2],po1[2]],c=bgr_)
        # ax.auto_scale_xyz([-2,1], [0.25,3.25], [-1,1])
        # plt.savefig()
        # plt.draw()
        # plt.pause(0.1)
        # ax.cla()
        
        cv2.putText(pic3D, text=f"Index_{ind}", org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,color=bgr[5], thickness=5)
        Dr_11.append(pic3D)
        Dr_11.append(np.ones((65, 1920, 3),np.uint8)*255)
        Dr_1_1 = np.concatenate((Dr_11[0],Dr_11[1]),axis=0)
        Dr_1_1 = np.concatenate((Dr_1_1,Dr_11[2]),axis=0)
        Dr_1.append(Dr_1_1)
        Dr_1.append(np.ones((1570, 92, 3),np.uint8)*255)
        
        for c in range(5):
            imgpath = os.path.join(Bathpath, campath_[c], img_path[ind])
            pic = cv2.imread(imgpath)
            indpoints.append([]) 
            imgs = []
            scales = []
            centers = []
            for p,ps in enumerate(box_censcl[ind][c]):
                center = ps[0][0]
                scale = ps[0][1]
                bbox_ = ps[1][0]
                rid = ps[1][1]
                if len(app_) < rid+1:
                    app_.append(0)
                    nwrong.append([rid,0])
                
                img, norm_img = process_image(imgpath, np.array(center), scale, input_res=constants.IMG_RES)
                with torch.no_grad():
                    pred_rotmat, pred_betas, pred_camera = p_net(norm_img.to(device))
                    pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                    pred_vertices = pred_output.vertices
                camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
                camera_translation = camera_translation[0].cpu().numpy()
                pred_vertices = pred_vertices[0].cpu().numpy()
                pose_points = np.dot(J_re_arr, np.array(pred_vertices))  #24x3
                img = img.permute(1,2,0).cpu().numpy()
                img_shape = renderer(pred_vertices, camera_translation, np.zeros_like(img))
                camera_translation[0] *= -1.  
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = camera_translation
                pose_xyz1 = np.ones((1,24,4))
                pose_xyz1[...,:3] = pose_points  #pose_3D
                camera_matrix = camera_pose[:3,:]
                pose_xy1 = (camera_matrix @ pose_xyz1[..., None])[...,0]
                pose_xy = pose_xy1[..., :-1][0]
                indpoints[c].append([[], rid])  
                if len(cpoints) < rid+1: 
                    for ins in range(rid+1-len(cpoints)):
                        cpoints.append([])
                cpoints[rid].append([[],c])# ps-[[24[x,y]], c]
                app_[rid] = 1  #state
                for jo in pose_xy:  
                    x = (jo[0]+1)*scale*100 + center[0]-scale*100
                    y = (jo[1]+1)*scale*100 + center[1]-scale*100
                    x_ = x.tolist()
                    y_ = y.tolist()
                    indpoints[c][p][0].append([x_,y_])    #cam-ps- [[24[x,y]], rid]
                    bgr_ = (bgr[rid][2]/255, bgr[rid][1]/255, bgr[rid][0]/255)
                    cv2.circle(pic, (int(x),int(y)), 12, bgr[rid], -1)
                    cpoints[rid][-1][0].append([x_,y_])
                for poid in config.POINTS_ID: 
                    po1,po2 = poid
                    x1,y1 = pose_xy[po1]
                    x2,y2 = pose_xy[po2]
                    x1 = (x1+1)*scale*100 + center[0]-scale*100
                    y1 = (y1+1)*scale*100 + center[1]-scale*100
                    x2 = (x2+1)*scale*100 + center[0]-scale*100
                    y2 = (y2+1)*scale*100 + center[1]-scale*100
                    cv2.line(pic, (int(x1),int(y1)), (int(x2),int(y2)), bgr[rid], 4)
            cv2.putText(pic, text=f"cam_{c}", org=(30,40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=bgr[5], thickness=3)
            if c ==0:
                Dr_13 = []
                Dr_13.append(pic)
            if c == 1:
                Dr_13.append(np.ones((18, 1032, 3),np.uint8)*255)
                Dr_13.append(pic)
                Dr_1_3 = np.concatenate((Dr_13[0],Dr_13[1]), axis=0)
                Dr_1_3 = np.concatenate((Dr_1_3,Dr_13[2]), axis=0)
                Dr_1.append(Dr_1_3)
                Dr_1.append(np.ones((1570, 14, 3),np.uint8)*255)
                sh1 = Dr_1[0]

                for n,d in enumerate(Dr_1):
                    if n > 0:
                        sh1 = np.concatenate((sh1,d), axis=1)
                Draw5.append(sh1)
                Draw5.append(np.ones((20, 3150, 3),np.uint8)*255)
            if c == 2:
                Dr_3 = []
                Dr_3.append(np.ones((776, 10, 3),np.uint8)*255)
                Dr_3.append(pic)
            if c == 3:
                Dr_3.append(np.ones((776, 15, 3),np.uint8)*255)
                Dr_3.append(pic)
            if c == 4:
                Dr_3.append(np.ones((776, 15, 3),np.uint8)*255)
                Dr_3.append(pic)
                Dr_3.append(np.ones((776, 14, 3),np.uint8)*255)
                sh2 = Dr_3[0]
                for n,d in enumerate(Dr_3):
                    if n > 0:
                        sh2 = np.concatenate((sh2,d), axis=1)
                Draw5.append(sh2)
                Draw5.append(np.ones((14, 3150, 3),np.uint8)*255)
        
        show = Draw5[0]
        for n,d in enumerate(Draw5):
            if n > 0:
                show = np.concatenate((show,d), axis=0)
        showView = cv2.resize(show,(1050, 800))
        cv2.imshow("show", showView)
        cv2.waitKey(1)
