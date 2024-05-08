#import
#!pip install ultralytics
import ultralytics
from ultralytics import YOLO
import sys
import os
import csv
from PIL import Image

#load the custom model
model = YOLO(sys.argv[1])
#get predicted bounding box data
data_test = sys.argv[2]
results = model.predict(source=data_test, conf=float(sys.argv[3]), imgsz=2000, iou=float(sys.argv[4]), max_det=1000)

fichiers = sorted(os.listdir(data_test))
#file for stats
fichier_existe = sys.argv[5]
with open(fichier_existe, mode='a', newline='') as fichier_csv:
    writer = csv.writer(fichier_csv,delimiter=';')
    #headers
    if not fichier_existe:
        writer.writerow(["method","model","dir","conf","iou","imgsz","max_det","path","pixx","pixy","nb","area_tot","w_mean","h_mean"])
    #for every image
    for i in range(len(fichiers)):

        path = fichiers[i]
        #determine number of pixels
        img_temp = Image.open(os.path.join(data_test,path))
        pix_x, pix_y = img_temp.size
        #number of insects detected
        nb = len(results[i])
        w_tot = 0
        h_tot = 0
        area_tot = 0
        #for every detection
        for j in range(nb):
            #determine w and h
            w = int(results[i][j].boxes.xywh[0][2])
            h = int(results[i][j].boxes.xywh[0][3])
            w_tot += w
            h_tot += h
            area_tot += w*h
        #compute mean
        w_mean = w_tot/nb
        h_mean = h_tot/nb
        #write new line with all infos
        writer.writerow(["YOLO",sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],2000,1000,path,pix_x,pix_y,nb,area_tot,w_mean,h_mean])
