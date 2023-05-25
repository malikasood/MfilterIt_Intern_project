import glob
import os
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
file = 'C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/VS CODE Scripts/flowers'
ext = ('jpeg', 'jpg')
#for f in glob.glob(file + "/*.jpg" or file + "/*.jpeg"):
for f in os.listdir(file):
    if f.endswith(ext):
        
        results =  model(f)
        print(f,results)
    else:
        continue


 

