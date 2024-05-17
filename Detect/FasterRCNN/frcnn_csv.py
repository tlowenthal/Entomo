import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import os
import sys
import csv

# Charger le modèle enregistré
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,box_detections_per_img=1000)
num_classes = 2  # Nous avons deux classes : background et objet
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()

# Définir les labels de classe
class_labels = {1: "insect"}

# Définir la fonction de prédiction
def predict(image_path, model):
    image = cv2.imread(image_path)
    #image = Image.open(image_path).convert("RGB")
    #image = image.transpose(Image.ROTATE_180)
    # Transformer l'image en tenseur
    image_tensor = F.to_tensor(image)
    # Ajouter une dimension pour le batch
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

# Dossier contenant les images de test
test_dir = sys.argv[2]
fichier_existe = sys.argv[5]
with open(fichier_existe, mode='a', newline='') as fichier_csv:
    writer = csv.writer(fichier_csv,delimiter=';')
    #headers
    if not fichier_existe:
        writer.writerow(["method","model","dir","conf","iou","imsz","max_det","path","pixx","pixy","nb","area_tot","w_mean","h_mean"])
        
# Boucle à travers les images de test
    for file in sorted(os.listdir(test_dir)):
        image_path = os.path.join(test_dir, file)
        if os.path.isfile(image_path):
            # Prédiction sur l'image
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
            #for every detection
            #image = cv2.imread(image_path)
            # Dessiner les boîtes englobantes prédites sur l'image
            for box_idx in filtered_boxes:
                box = prediction[0]["boxes"][box_idx]
                score = prediction[0]["scores"][box_idx]
                label = prediction[0]["labels"][box_idx]
                if score > float(sys.argv[3]):  # Seuil de confiance
                    box = [int(coord) for coord in box]
                    #cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
                    w = box[2]-box[0]
                    h = box[3]-box[1]
                    w_tot += w
                    h_tot += h
                    area_tot += w*h
                    nb2 += 1
                    
            if(nb == nb2):
                print("ok")
            else:
                print(nb2,nb)
            w_mean = w_tot/nb2
            h_mean = h_tot/nb2
            #write new line with all infos
            writer.writerow(["FasterRCNN",sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],0, 1000,image_path,pix_x,pix_y,nb2,area_tot,w_mean,h_mean])
                        
