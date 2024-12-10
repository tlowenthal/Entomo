#import
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys

image_path = [] #list of image names contained in the folder
boxes = [] #list of bounding box coordinates
labels_list = []

dir1 = sys.argv[1]
for file in sorted(os.listdir(dir1)):
    path = os.path.join(dir1, file)
    if os.path.isfile(path):
        image_path.append(file)
print(image_path)

dir2 = sys.argv[2]
count_img=0
for file in sorted(os.listdir(dir2)):
    path = os.path.join(dir2, file)
    print(path)
    if os.path.isfile(path):
        img_temp = Image.open(os.path.join(dir1,image_path[count_img]))
        pix_x, pix_y = img_temp.size
        temp_boxes = []
        temp_labels = []
        with open(path, 'r') as txt:
            lines = txt.readlines()
        for line in lines:
            elements_line = line.strip().split()
            class_id = int(elements_line[0])
            x_center = float(elements_line[1])*pix_x
            y_center = float(elements_line[2])*pix_y
            width = float(elements_line[3])*pix_x
            height = float(elements_line[4])*pix_y
            
            x1 = round(x_center - width/2)
            y1 = round(y_center - height/2)
            x2 = round(x_center + width/2)
            y2 = round(y_center + height/2)
            
            elements = [x1,y1,x2,y2]
            temp_boxes.append(elements)

            if class_id == 0:
                temp_labels.append(1)
            elif class_id == 1:
                temp_labels.append(0)

        boxes.append(temp_boxes)
        labels_list.append(temp_labels)
        count_img += 1

class Cust(torch.utils.data.Dataset):
    def __init__(self, image_path, boxes, labels_list, indexes):
        self.image_path = image_path
        self.boxes = boxes
        self.labels_list = labels_list
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,indx):
        img_name = self.image_path[self.indexes[indx]]
        img_boxes = np.array(self.boxes[self.indexes[indx]]).astype('float')
        img_labels = np.array(self.labels_list[self.indexes[indx]]).astype('int64')

        #img = Image.open("./f_coleo/img_train/"+img_name).convert('RGB')
        img = cv2.imread(os.path.join(dir1, img_name))
        #img = img.transpose(Image.ROTATE_180)
        #labels = torch.ones((len(img_boxes)), dtype=torch.int64)
        target = {}
        target["boxes"] = torch.tensor(img_boxes)
        target["labels"] = torch.tensor(img_labels)

        return T.ToTensor()(img),target

def cust_coll(data):
    return data

train, val = train_test_split(range(len(image_path)),test_size=0.2)

train_dl = torch.utils.data.DataLoader(Cust(image_path,boxes,labels_list, train),batch_size=16,shuffle=True,collate_fn=cust_coll,pin_memory=True if torch.cuda.is_available() else False)

val_dl = torch.utils.data.DataLoader(Cust(image_path,boxes, labels_list,val),batch_size=8,shuffle=True,collate_fn=cust_coll,pin_memory=True if torch.cuda.is_available() else False)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True,box_detections_per_img=1000)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
num_ep = 100

model.to(device)
for ep in range(num_ep):
    ep_loss = 0
    for data in train_dl:
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(device))
            targ = {}
            targ['boxes'] = d[1]["boxes"].to(device)
            targ['labels'] = d[1]["labels"].to(device)
            targets.append(targ)
        loss_dict = model(imgs,targets)
        loss = sum(v for v in loss_dict.values())
        ep_loss += loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(ep_loss)
#save model
torch.save(model.state_dict(), sys.argv[3])
