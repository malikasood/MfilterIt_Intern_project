import torch
import glob
import os
import cv2

#loading the model for testing
model = torch.hub.load('C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/yolov5', 'custom', path=r'C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/best_food_labels.pt', force_reload=True,source = 'local')
file = 'C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/Video/ad_video'

for f in glob.glob(file +"/*"):
    img = cv2.imread(f)
    image_name = f.split("/")[-1].split('\\')[-1]
    print(image_name)

    results = model(f)
    results = eval(results.pandas().xyxy[0].to_json(orient="records"))
    print(results)
    food_label= []
#up until here the results give all the various attributes of the image after testing ie coordinates,confidence,class and name of the class

#to get the bounding box for labels that have been detected in the previous step,but using condition in confidence  
    if results!= []:
      for k in results:
        if k['name'] == 'veg' and k['confidence'] >= 0.65:
            food_label.append(k['name'])
            food_label.append(k['confidence'])
        elif k['name'] == 'non_veg' and k['confidence'] >= 0.75:
            food_label.append(k['name'])
            food_label.append(k['confidence'])

#printing coordinates of a rectangle(integer values) 
        xmin = int(k['xmin'])
        ymin = int(k['ymin'])
        xmax = int(k['xmax'])
        ymax = int(k['ymax'])
        print (xmin,ymin,xmax,ymax)

#for creating rectangle using cv2 library

        start_point = (xmin,ymin)
        end_point = (xmax,ymax)
        color = (0,255,255)
        thickness=2
        image = cv2.rectangle(img, start_point, end_point, color, thickness)

#to show and save the frames 
#cv2.imread() for reading the image to a variable and cv2.imshow() to display the image in a separate window.
#cv2.imwrite()
     
        cv2.imshow('Bounding box images',image)
        cv2.imwrite("C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/TRAINED DATA food_label_detection/images/train/ad_bounding_images/"+ image_name,image)

    print(food_label)

