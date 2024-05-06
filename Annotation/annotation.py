
#import
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout ,  QFileDialog, QLabel, QLineEdit
from PyQt5 import QtCore
import cv2
import numpy as np
import sys
import os
import shutil
import csv

#definition of a maximum image display size for easy GUI use
MAX_WIDTH=1500
MAX_HEIGHT=1000
MAX_WIDTH2=1800
MAX_HEIGHT2=1200

#to leave while loop
proc = True

def nothing(x):
    pass

def close_project():
    sys.exit(app.exec_())

#def to calibrate parameters detection
def load_img(img_path):
    #global vars to use in others defs
    global contr
    global proc
    
    global hMin 
    global sMin 
    global vMin 
    global hMax
    global sMax 
    global vMax
    global blur
    global erodeX
    global erodeY
    global sizeMin
    global sizeMax
    global marginX 
    global marginY
    global offsetX 
    global offsetY 
    global contt

    #load visualization image + size, create an hsv version
    image = cv2.imread(img_path)
    height, width, channels = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #resize if the maximum size is exceeded
    if width>MAX_WIDTH:
        #determine factor of reduction
        f1 = MAX_WIDTH / width
        f2 = MAX_HEIGHT / height
        f = min(f1, f2)
        dim = (int(width * f), int(height * f))
        hsv = cv2.resize(hsv, dim)
        image = cv2.resize(image,dim)
    
    #create a blank window
    cv2.namedWindow('image')

    #create cursors to be able to modify parameters
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('Blur', 'image', 0, 150, nothing)
    cv2.createTrackbar('ErodeX', 'image', 0, 1000, nothing)
    cv2.createTrackbar('ErodeY', 'image', 0, 1000, nothing)
    cv2.createTrackbar('AreaMin', 'image', 0, 10000, nothing)
    cv2.createTrackbar('AreaMax', 'image', 0, 1000000, nothing)
    cv2.createTrackbar('MarginX','image',0,100,nothing)
    cv2.createTrackbar('MarginY','image',0,100,nothing)
    cv2.createTrackbar('OffsetX','image',0,200,nothing)
    cv2.createTrackbar('OffsetY','image',0,200,nothing)
    cv2.createTrackbar('HierarchyLevel','image',0,1,nothing)
    
    #default values
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)
    cv2.setTrackbarPos('AreaMin', 'image', 10000)
    cv2.setTrackbarPos('AreaMax', 'image', 1000000)
    cv2.setTrackbarPos('AreaMin', 'image', 10000)
    cv2.setTrackbarPos('AreaMax', 'image', 100000)
    cv2.setTrackbarPos('OffsetX', 'image', 100)
    cv2.setTrackbarPos('OffsetY', 'image', 100)

    while(1):
        #takes current cursor positions as parameter values
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')
        blur = cv2.getTrackbarPos('Blur', 'image')
        erodeX = cv2.getTrackbarPos('ErodeX', 'image')
        erodeY = cv2.getTrackbarPos('ErodeY', 'image')
        sizeMin = cv2.getTrackbarPos('AreaMin', 'image')
        sizeMax = cv2.getTrackbarPos('AreaMax', 'image')
        marginX = cv2.getTrackbarPos('MarginX','image')
        marginY = cv2.getTrackbarPos('MarginY','image')
        offsetX = cv2.getTrackbarPos('OffsetX','image')
        offsetY = cv2.getTrackbarPos('OffsetY','image')  
        contt = cv2.getTrackbarPos('HierarchyLevel','image')
          
        #threshold bounds
        lower = np.array([hMin, sMin, vMin], np.uint8)
        upper = np.array([hMax, sMax, vMax], np.uint8)

        #application of the threshold
        mask = cv2.inRange(hsv, lower, upper)
        
        #kernel size must be odd for median blur and erode
        #apply median blur
        if(blur%2==1):    
            med_blur = cv2.medianBlur(mask,blur)  
        else:
            med_blur = cv2.medianBlur(mask,blur+1)
        
        if(erodeX%2==1):
            erode_horiz = erodeX
        else:
            erode_horiz = erodeX+1
        if(erodeY%2==1):
            erode_vertic = erodeY  
        else:
            erode_vertic = erodeY+1
        
        #apply erosion 
        kernel = np.ones((erode_vertic, erode_horiz), np.uint8)
        erode = cv2.dilate(med_blur, kernel)
        #add a contour to the edge to close all contours
        erode_with_border = cv2.copyMakeBorder(erode,1,1,1,1, cv2.BORDER_CONSTANT, value=255)
        #apply canny to detect edges
        edges = cv2.Canny(erode_with_border, 0, 3)
        #dilate edges a bit to avoid discontinuity
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        edges_dilated = cv2.erode(edges_dilated, np.ones((3, 3), np.uint8))  
 
        #list every contours
        if contt == 0:
            contours, _ = cv2.findContours(edges_dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            filtered_contours = contours
        else:
            contours, hierarchy = cv2.findContours(edges_dilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            #create an empty list to store the indices of the filtered contours
            filtered_contours_indices = []
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:  #if the outline has a level 1 parent
                    parent_index = hierarchy[0][i][3]
                    if hierarchy[0][parent_index][3] != -1:
                        parent_index2 = hierarchy[0][parent_index][3]
                        if hierarchy[0][parent_index2][3] == -1:
                                filtered_contours_indices.append(i)

            #create a list of filtered contours from indices
            filtered_contours = [contours[i] for i in filtered_contours_indices]

        #put in global var
        contr = filtered_contours
        
        #copy original and eroded image for visualization
        out_visu = image.copy()
        
        erode_visu = erode_with_border.copy()
        
        #each contour between sizeMin and sizeMax is taken into account 
        for cont in filtered_contours:
            area = cv2.contourArea(cont)
            if area > sizeMin and area < sizeMax:  
                #bounding box coordinates    
                x,y,w,h = cv2.boundingRect(cont)
                #if the contour is too long, it is rejected
                if w>0.75*width or h>0.75*height:
                    continue
                    
                #apply a margin on each side
                offset_xtrue = 100-offsetX
                offset_ytrue = 100-offsetY
                x_i = max(0,x-marginX-offset_xtrue)
                x_f = min(width,x+w+marginX-offset_xtrue)
                y_i = max(0,y-marginY+offset_ytrue)
                y_f = min(height,y+h+marginY+offset_ytrue)
                
                #visualization of bounding boxes on original and eroded image 
                cv2.rectangle(erode_visu, (x_i,y_i), (x_f,y_f), (0, 255, 0), 1)
                cv2.rectangle(out_visu, (x_i,y_i), (x_f,y_f), (255, 0, 0), 1)
        
        
        cv2.imshow('image3', out_visu)
        cv2.imshow('image2', erode_visu)
        cv2.moveWindow('image', 20, 700)
        
        cv2.moveWindow('image3', 1000, 500)
        cv2.moveWindow('image2', 1000, 50)
        
        #quit the GUI when "q" is pressed
        if cv2.waitKeyEx(10) & 0xFF == ord('q'):
            break
    
        if proc == False :
            proc = True
            break

    cv2.destroyAllWindows()
    visu_ann(img_path_visu,output_file_path)



#def to select an image for visualization
def choose_img(x):
    #global variables for path to folder and visualization image
    
    global img_path_visu
    
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    img_path, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
    print("IMAGE SELECTED")
    #save the path to the image and its folder
    if img_path:
        img_path_visu = img_path
        print(img_path_visu)
        load_img(img_path_visu)


def suppr(x):
    global proc
    global output_file_path
    image = cv2.imread(img_path_visu)
    height, width, channels = image.shape
    if width>MAX_WIDTH:
        #determine factor of reduction
        f1 = MAX_WIDTH / width
        f2 = MAX_HEIGHT / height
        f = min(f1, f2)
        dim = (int(width * f), int(height * f))
        image = cv2.resize(image,dim)
    height, width, channels = image.shape
    data_annotation = "Annotation_temp"
    if not os.path.exists(data_annotation):
            os.makedirs(data_annotation)
    #set the output annotation file path
    image_filename_sp, _ = os.path.splitext(os.path.basename(img_path_visu))
    image_filename_txt = image_filename_sp + ".txt"
    output_file_path = os.path.join(data_annotation,image_filename_txt)
    
    fichier_existe = 'out.csv'
    with open(fichier_existe, mode='a', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv,delimiter=';')

        #write headers if the file has just been created
        if not fichier_existe:
            writer.writerow(["H_min", "S_min", "V_min","H_max", "S_max", "V_max","Blur","Erode_x","Erode_y","Area_min","Area_max","Margin_x","Margin_y","Offset_x","Offset_y","HierarchyLevel","Path"])

        #write a new line with the values
        writer.writerow([hMin,sMin,vMin,hMax,sMax,vMax,blur,erodeX,erodeY,sizeMin,sizeMax,marginX,marginY,offsetX,offsetY,contt,img_path_visu])
    #open annotation file
    with open(output_file_path, 'w') as output_file:

        #each contour between sizeMin and sizeMax is taken into account 
        for contour in contr:
            area = cv2.contourArea(contour)
            if area > sizeMin and area < sizeMax:  
                #bounding box coordinates         
                x,y,w,h = cv2.boundingRect(contour)
                #if the contour is too long, it is rejected
                if w>0.75*width or h>0.75*height:
                    continue
                    
                offset_xtrue = 100-offsetX
                offset_ytrue = 100-offsetY
                x_i_det = max(0,x-marginX-offset_xtrue)
                x_f_det = min(width,x+w+marginX-offset_xtrue)
                y_i_det = max(0,y-marginY+offset_ytrue)
                y_f_det = min(height,y+h+marginY+offset_ytrue)
                
                #normalized coordinates
                x_center = (x_i_det + (x_f_det-x_i_det)/ 2) / width
                y_center = (y_i_det + (y_f_det-y_i_det)/ 2) / height
                width_normalized = (x_f_det-x_i_det) / width
                height_normalized = (y_f_det-y_i_det) / height
               
                #write in the anntation file
                output_file.write(f"{0} {x_center} {y_center} {width_normalized} {height_normalized}\n")

    proc = False

def visu_ann(image_path, annotation_path):
    #load image
    global visu
    global nbb
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    if width>MAX_WIDTH2:
        #determine factor of reduction
        f1 = MAX_WIDTH2 / width
        f2 = MAX_HEIGHT2 / height
        f = min(f1, f2)
        dim = (int(width * f), int(height * f))
        image = cv2.resize(image,dim)
    height, width, _ = image.shape

    #read annotations
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()

    for idx, annotation in enumerate(annotations):
        #split values
        values = annotation.strip().split(' ')
        label = int(values[0])
        x_center = float(values[1]) * width
        y_center = float(values[2]) * height
        box_width = float(values[3]) * width
        box_height = float(values[4]) * height

        #calculate bounding box corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        transparency = 0.2
        overlay = image.copy()
        #draw a solid rectangle on a transparent layer
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
        image = cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)

        #draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        text_size = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1 + int((box_width - text_size[0]) / 2)
        text_y = y1 - 10 + int((box_height + text_size[1]) / 2)
        cv2.putText(image, str(idx), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
       
       
    total_detections = len(list(enumerate(annotations)))
    nbb = total_detections
    cv2.putText(image, "Total Detections: " + str(total_detections), (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    #show image with bounding boxes and IDs
    visu = image
    cv2.imshow('Annotations', image)
    cv2.moveWindow('Annotations', 800, 50)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dell(x):
    cv2.destroyAllWindows()
    global output_file_path
    valeurs = line_edit.text().split(",")
    valeurs = list(map(int, valeurs))
    todl = set(valeurs)

    with open(output_file_path, 'r') as f:
        lignes = f.readlines()

    lignes_filtrees = [ligne for index, ligne in enumerate(lignes) if index not in todl]

    with open("out.txt", 'w') as f:
        f.writelines(lignes_filtrees)

    #replace input file with output file
    shutil.move("out.txt", output_file_path)
    visu_ann(img_path_visu,output_file_path)
    
def addd(x):

    cv2.destroyAllWindows()
    
    image = cv2.imread(img_path_visu)
    height, width, channels = image.shape
    if width>MAX_WIDTH2:
        #determine factor of reduction
        f1 = MAX_WIDTH2 / width
        f2 = MAX_HEIGHT2 / height
        f = min(f1, f2)
        dim = (int(width * f), int(height * f))
        image = cv2.resize(image,dim)
    height, width, _ = image.shape

    #creating a new window with a custom size
    cv2.namedWindow("Select ROIs", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROIs", MAX_WIDTH2, MAX_HEIGHT2)
    cv2.moveWindow("Select ROIs", 800, 50)
    

    #launch of the ROI selector
    ROIs = cv2.selectROIs("Select ROIs", visu, False, False)
  
    cv2.destroyAllWindows()
    
    uniq = set()
    for rect in ROIs:
        global w
        global h
        
        #take coordinates
        x_i = rect[0]
        y_i = rect[1]
        w = rect[2]
        h = rect[3]
        
        x_f = x_i+w
        y_f = y_i+h
        
        
        #normalized coordinates
        x_center = (x_i + (x_f-x_i)/ 2) / width
        y_center = (y_i + (y_f-y_i)/ 2) / height
        width_normalized = (x_f-x_i) / width
        height_normalized = (y_f-y_i) / height
        
        uniq.add(f"{0} {x_center} {y_center} {width_normalized} {height_normalized}\n")
        
    if uniq :
        with open(output_file_path, 'a') as output_file: 
            for u in uniq:
                output_file.write(u)
                
    visu_ann(img_path_visu,output_file_path)

class ImageClicker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = []

        self.load_image()

    def load_image(self):
        #load image
        image = cv2.imread(self.image_path)
        height, width, channels = image.shape
        if width>MAX_WIDTH2:
            #determine factor of reduction
            f1 = MAX_WIDTH2 / width
            f2 = MAX_HEIGHT2 / height
            f = min(f1, f2)
            dim = (int(width * f), int(height * f))
            image = cv2.resize(image,dim)



        self.image = visu

        cv2.namedWindow('click')
        cv2.moveWindow('click', 800, 50)
        
        cv2.setMouseCallback('click', self.on_click)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
     


    def run(self):
        cv2.imshow('click', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_points(self):
        return self.points

def adddc(x):
    cv2.destroyAllWindows()
    clicker = ImageClicker(img_path_visu)
    clicker.run()
    points = clicker.get_points()
    
    
    uniq = set()
    for p in points:
        
        
        #normalized coordinates
        x_center = p[0] / MAX_WIDTH2
        y_center = p[1] / MAX_HEIGHT2
        width_normalized = w / MAX_WIDTH2
        height_normalized = h / MAX_HEIGHT2
        
        uniq.add(f"{0} {x_center} {y_center} {width_normalized} {height_normalized}\n")
        
    if uniq :
        with open(output_file_path, 'a') as output_file: 
            for u in uniq:
                output_file.write(u)
                
    visu_ann(img_path_visu,output_file_path)


def endd(x):
    fichier_existe = 'outnb.csv'
    with open(fichier_existe, mode='a', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv,delimiter=';')


        #write headers if the file has just been created
        if not fichier_existe:
            writer.writerow(["Nb","Path"])

        #write a new line with the values
        writer.writerow([nbb,img_path_visu])
    
    dossier_destination = "Annotation_final"
    if not os.path.exists(dossier_destination):
        os.makedirs(dossier_destination)

    #copy the file to the destination folder
    shutil.copy(output_file_path, dossier_destination)
    cv2.destroyAllWindows()

   
#create actions window with buttons
app = QApplication([])
window = QWidget()
window.setWindowTitle("Insect counter V3")
window.setMinimumWidth(600)
layout = QVBoxLayout()
but_img=QPushButton('1 : Choose image')
layout.addWidget(but_img)
but_img.clicked.connect(choose_img)

but_frz=QPushButton('2 : Freeze parameters')
layout.addWidget(but_frz)
but_frz.clicked.connect(suppr)
label = QLabel("List elements to delete")
line_edit = QLineEdit()
layout.addWidget(label)
layout.addWidget(line_edit)

but_del=QPushButton('3 : Delete detections')
layout.addWidget(but_del)
but_del.clicked.connect(dell)

but_add=QPushButton('4-1 : Add detections (ROI)')
layout.addWidget(but_add)
but_add.clicked.connect(addd)

but_addc=QPushButton('4-2 : Add detections (click)')
layout.addWidget(but_addc)
but_addc.clicked.connect(adddc)

but_end=QPushButton('5 : Save all')
layout.addWidget(but_end)
but_end.clicked.connect(endd)

window.setLayout(layout)
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
window.move(20,1)
window.show()
app.exec()
