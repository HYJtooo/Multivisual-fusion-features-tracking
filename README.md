# Multivisual-fusion-features-tracking
Under multi-vision, feature extraction and correlation are carried out based on the similarity of pedestrian features, and multi-target pedestrian tracking in dense pedestrian scene is completed through GMM-based feature pool update, pedestrian addition and recurrence judgment and other supplementary verification operations.

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
Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from: https://pan.baidu.com/s/1nuWalVLYrkMbiye-6kkzIQ. code: ef6g  
Save `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in `log_Shelf/`  

## Run
run example `demo_.py`  

## Qualitative Result
part1:   
![gif1](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/HYJtooo-patch-1/part1-gif.gif)  

part2:  
![gif2](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/HYJtooo-patch-1/part2-gif.gif)  


![gif](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/main/result1.gif)  

![gif2](https://github.com/HYJtooo/Multivisual-fusion-features-tracking/blob/main/result2.gif)
