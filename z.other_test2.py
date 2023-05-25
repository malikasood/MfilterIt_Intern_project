import torch
import glob
import os
# Model
model = torch.hub.load(r'C:\Users\Malika Sood\Desktop\Intern\Mfilterit\Project\yolov5', 'custom', path=r'C:\Users\Malika Sood\Desktop\Intern\Mfilterit\Project\best_food_labels.pt', force_reload=True,source = 'local')
file = r'C:\Users\Malika Sood\Desktop\Intern\Mfilterit\Project\test2'

for f in glob.glob(file +"/*"):
    results = model(f)
    results = eval(results.pandas().xyxy[0].to_json(orient="records"))
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

  