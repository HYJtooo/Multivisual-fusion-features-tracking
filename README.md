# Multivisual-fusion-features-tracking
In multiple views, 3D pose point tracking is implemented by a tracking module and a pose detection module. Based on fusion features designed an extraction model, dynamic threshold strategy, GMM feature pool update model and verification model. After 2D regression of the pose points through the SMPL basic model, the real camera parameters are projected to complete the accurate and complete 3D pose point tracking of pedestrians.

## Environment
Ubuntu 18.04  
Python 3.6.4  
cuda 11.1  
cudnn 8.0.5  
Pytorch 1.10.0 + cu111

## Dataset
Download dataset `Shelf` from: https://pan.baidu.com/s/1z9vOfW2klU2PK2qy0sajAg. code: sh4e  
Save dataset in the root  

## Model
Download `model.pth` from: https://pan.baidu.com/s/1nuWalVLYrkMbiye-6kkzIQ. code: ef6g  
Save `model.pth` in `log_Shelf/`  
Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from: https://pan.baidu.com/s/1MraSJmLwRMVJcKx0GAvDTA. code: sh4e.   
Save `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in `log_Shelf/`  

## Run
run example `demo_.py`  

## Qualitative Results

### Part1:     
Our work can effectively identify pedestrians when newly added and partial occlusion occur.   
![gif1](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/HYJtooo-patch-1/part1-gif.gif)  

### Part2:  
Our work has satisfactory tracking capabilities for pedestrians who reappear after completely disappearing from views.   
![gif2](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/HYJtooo-patch-1/part2-gif.gif)  

## Quantitative Results

### The Number of Failed Tracking
To demonstrate the algorithm's ability to control the overall tracking error.   
![pic1](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/HYJtooo-patch-1/failedtracking.svg)  

### The Success Rate of 3D Pose Points
To reflect the success rate of the pose points tracking within each pedestrian.   
![pic2](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/HYJtooo-patch-1/pointssuccess.svg)
