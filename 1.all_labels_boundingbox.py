import torch
import glob
import os
import cv2

# Model
model= torch.hub.load('C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/yolov5', 'custom', path='C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/best_food_labels.pt', force_reload=True,source = 'local')
file = 'C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Project/test_food_labels'

#for accessing all the images within the folder 
for f in glob.glob(file +"/*"):
    img = cv2.imread(f)
    image_name = f.split("/")[-1].split('\\')[-1]
    print(image_name)


    results = model(f)
    results = eval(results.pandas().xyxy[0].to_json(orient="records"))
    print(results)
    food_label= []
    
    if results!= []:
      for k in results:
        if k['name'] == 'veg' and k['confidence'] >= 0.5:
            food_label.append(k['name'])
            food_label.append(k['confidence'])
        elif k['name'] == 'non_veg' and k['confidence'] >= 0.75:
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

 #to show and save the frames         
        cv2.imshow('bounding box',image)
        print("C:/Users/Malika Sood/Desktop/InternMfilteritProject/TRAINED DATA food_label_detection/images/train/test_food_labels_images/"+ image_name)
        cv2.imwrite("C:/Users/Malika Sood/Desktop/InternMfilteritProject/TRAINED DATA food_label_detection/images/train/test_food_labels_images/"+ image_name,image)
     
        
    print(food_label)
