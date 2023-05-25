import torch
import glob

#files = 'image/'

# Model
 model = torch.hub.load(r'C:\Users\Malika Sood\Desktop\Intern\Mfilterit\Project\yolov5', 'custom', path=r'C:\Users\Malika Sood\Desktop\Intern\Mfilterit\Project\best_food_labels.pt', force_reload=True,source = 'local')

#for f in glob.glob(files+"/*"):
results = model(r'C:\Users\Malika Sood\Desktop\Intern\Mfilterit\Project\food_label_detection\images\train\veg133.jpg')
print(results.pandas().xyxy[0])
#      .to_json(orient="records"))

******************************

import torch
import glob

#files = 'image/'

# Model
model = torch.hub.load("D:/Project/yolov5", 'custom', path='best_food_labels.pt', force_reload=True,source = 'local')


#for f in glob.glob(files+"/*"):

results = model("D:/Project/veg13.jpg")
print(results.pandas().xyxy[0])
#      .to_json(orient="records"))

#test
results = model("C:/Users/Malika Sood/Downloads/ask_chkra/ashok_chakra/ashok_chakra")
