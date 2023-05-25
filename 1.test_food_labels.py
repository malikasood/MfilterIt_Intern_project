import torch
import glob
import os
import cv2

# Model
model= torch.hub.load('C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/yolov5', 'custom', path='C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/best_food_labels.pt', force_reload=True,source = 'local')
new = 'C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/test_food_labels/02_Frozen snacks - 1.jpg'
img = cv2.imread(new)
#size= cv2.resize(img,(1280,1280))
#cv2.imwrite("abc.jpg",size)

#for accessing all the images within the folder 
'''for f in glob.glob(file +"/*"):
    img = cv2.imread(f)
    image_name = file.split("/")[-1]'''

results = model(img)
results = eval(results.pandas().xyxy[0].to_json(orient="records"))
print(results)
food_label= []

if results!= []:
    for k in results:
        food_label.append(k['name'])
        food_label.append(k['confidence'])
        xmin = int(k['xmin'])
        ymin = int(k['ymin'])
        xmax = int(k['xmax'])
        ymax = int(k['ymax'])
        print (xmin,ymin,xmax,ymax)

        # represents the top left corner of rectangle
        start_point = (xmin,ymin)

# represents the bottom right corner of rectangle
        end_point = (xmax,ymax)

# Blue color in BGR
        color = (255, 0, 0)
        thickness = 2

        image = cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.imshow('bounding box',image)

#to show and save the frames 
        cv2.imwrite("aa.jpg",image)
     
        
    print(food_label)







   
