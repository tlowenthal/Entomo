import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import os
import sys
import csv


#load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,box_detections_per_img=1000)
num_classes = 2  #two classes : background and objet
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()

#define label
class_labels = {1: "insect"}

#prediction function
def predict(image_path, model):
    image = cv2.imread(image_path)
    image_tensor = F.to_tensor(image)
    #add a dimension for the batch
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

#folder that contain test images
test_dir = sys.argv[2]
fichier_csv_arg = sys.argv[5]
fichier_existe = os.path.exists(fichier_csv_arg)
with open(fichier_csv_arg, mode='a', newline='') as fichier_csv:
    writer = csv.writer(fichier_csv,delimiter=';')
    #headers
    if not fichier_existe:
        writer.writerow(["method","model","dir","conf","iou","imsz","max_det","path","pixx","pixy","nb","area_tot","w_mean","h_mean","boxes"])
        
#for every image
    for file in sorted(os.listdir(test_dir)):
        image_path = os.path.join(test_dir, file)
        if os.path.isfile(image_path):
            
            #determine number of pixels
            img_temp = Image.open(image_path)
            pix_x, pix_y = img_temp.size
            
            prediction = predict(image_path, model)
            filtered_boxes = torchvision.ops.nms(prediction[0]["boxes"], prediction[0]["scores"], float(sys.argv[4]))
            #number of insects detected
            
            nb = len(filtered_boxes)
            nb2 = 0
            w_tot = 0
            h_tot = 0
            area_tot = 0
            coordinates = []
            #for every detection
      
            for box_idx in filtered_boxes:
                box = prediction[0]["boxes"][box_idx]
                score = prediction[0]["scores"][box_idx]
                label = prediction[0]["labels"][box_idx]
                if score > float(sys.argv[3]):  # Seuil de confiance
                    box = [int(coord) for coord in box]
                   
                    w = box[2]-box[0]
                    h = box[3]-box[1]
                    w_tot += w
                    h_tot += h
                    area_tot += w*h
                    nb2 += 1

                    x_min = box[0]
                    y_min = box[1]
                    x_max = box[2]
                    y_max = box[3]

                    coordinates.append((x_min,x_max,y_min,y_max))

                    
            if(nb == nb2):
                print("ok")
            else:
                print(nb2,nb)
            w_mean = w_tot/nb2
            h_mean = h_tot/nb2
            w_mean = float(f"{w_mean:.2f}")  
            h_mean = float(f"{h_mean:.2f}")
            #write new line with all infos
            writer.writerow(["FasterRCNN",sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],0, 1000,image_path,pix_x,pix_y,nb2,area_tot,w_mean,h_mean, coordinates])
                        
