import torch
import glob
import os
import cv2

# Model
model= torch.hub.load('C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/yolov5', 'custom', path='C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/best_food_labels.pt', force_reload=True,source = 'local')
file = 'C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/test_food_labels'
#img = cv2.imread(new)
#for accessing all the images within the folder 
for f in glob.glob(file +"/*"):
    #img = cv2.imread(f)
    #image_name = file.split("/")[-1]
    results = model(f)
    results = eval(results.pandas().xyxy[0].to_json(orient="records"))
    #print(results)
    food_label= []
    if results!= []:
        for k in results:
            if k['name'] == 'veg' and k['confidence'] >= 0.8:
                food_label.append(k['name'])
                food_label.append(k['confidence'])
            elif k['name'] == 'non_veg' and k['confidence'] >= 0.75:
                food_label.append(k['name'])
                food_label.append(k['confidence'])
    print(food_label)
