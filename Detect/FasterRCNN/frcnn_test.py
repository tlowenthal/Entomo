import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os
import sys

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

#for every image
for file in os.listdir(test_dir):
    image_path = os.path.join(test_dir, file)
    if os.path.isfile(image_path):

        prediction = predict(image_path, model)
        filtered_boxes = torchvision.ops.nms(prediction[0]["boxes"], prediction[0]["scores"], float(sys.argv[4]))
        image = cv2.imread(image_path)
        #draw bounding boxes
        for box_idx in filtered_boxes:
            box = prediction[0]["boxes"][box_idx]
            score = prediction[0]["scores"][box_idx]
            label = prediction[0]["labels"][box_idx]
            if score > float(sys.argv[3]):  # Seuil de confiance
                box = [int(coord) for coord in box]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
                
        #save image with bounding boxes
        cv2.imwrite(os.path.join(sys.argv[5], f"result_{file}"), image)
