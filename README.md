# AI in entomology
Welcome to this GitHub repository dedicated to artificial intelligence in entomology !

This repository houses a collection of tools designed to handle numerical data from entomological collections. Whether you're a researcher, entomologist, or data enthusiast, this toolbox provides a range of utilities tailored to analysis and management of insect-related data.

From stacking and annotation to the use of algorithm of detection, this toolbox offers a comprehensive suite of resources to support your work in the field of entomology.

Explore the repository, leverage the tools, and contribute to the advancement of entomological research with the power of data-driven insights.
## Tools Overview
### Image Stacking Tool

The Image Stacking Tool included in this repository offers a solution for merging multiple images together, particularly useful in entomological photography where depth of field can be an issue. This tool enables users to combine a series of photographs into a single, high-quality composite image, ensuring that all parts of the subject are in focus.
### Image Annotation Tool

Our Image Annotation Tool simplifies the process of labeling and annotating images, a fundamental step in training machine learning models for object detection tasks. With this tool, users can easily draw bounding boxes around insects within images, providing the necessary annotations for training robust detection algorithms.
### Insect Detection Algorithms

Within this repository, you'll find pre-trained models for two common orders of insects: Coleoptera (beetles) and Lepidoptera (butterflies and moths). These models are ready to use out of the box, allowing for quick and accurate detection of these insect groups within entomological box images.

Additionally, we provide resources and guidance on how to train your own custom detection models. Whether you're interested in detecting other insect orders, specific species, or unique characteristics within your entomological specimens, our repository offers the framework and tools to create tailored detection models to suit your research needs.

By offering pre-trained models as well as the flexibility to stack, annotate and create custom models , we aim to empower researchers and enthusiasts in their efforts to study and conserve insect biodiversity.

### Insect Classification Algorithm
This repository features a powerful Insect Classification Algorithm built on the ResNet architecture. Optimized for entomological research, this tool excels in identifying insect species based on high-resolution images.

The algorithm is capable of accurately classifying individual insects into taxonomic categories such as Order, Family, Genus, and Species. Pre-trained on a curated dataset of insect images, it offers reliable predictions out of the box. Additionally, users can fine-tune the model to classify custom species or adapt it to specific classification tasks, ensuring flexibility for diverse research needs.

## Use
### Image Stacking Tool

To stack photos of several boxes, you simply has to run this command :  
`python auto_stack.py <path to the folder> <number of pictures per zone>`  
where :
 - 'path to the folder' points to the folder containing all the boxes to be stacked. it must take the following form :
  
Folder  
   ├─ Box 1  
   │  &ensp;  ├─ Image 1  
   │  &ensp;  ├─ Image 2  
   │   &ensp; └─ ...  
   ├─ Box 2  
   │  &ensp;  ├─ Image 1  
   │  &ensp;  ├─ Image 2  
   │  &ensp;  └─ ...  
   └─ ...  
   Please organize the images so that those belonging to the same zone are ordered consecutively. 
 - 'number of pictures per zone' is the number of picture of the same zone taken at differents focus

### Image Annotation Tool
Lauch the GUI using this command :  
`python annotation.py`  

1. Choose the image that you want to annotate
Two windows will appear to visualize parameter calibration in order to automatically annotate as many insects as possible :
 - the minimum and maximum values of the HSV space (Hue,Saturation,Value) can be selected using the first 6 cursors
 - the following three sliders influence the contours of the positive zones of the HSV mask
 - the minimum and maximum area of the interior of a contour can be defined to be considered a detection using the following 2 cursus
 - the following four cursuers are used to adjust the size and position of bounding boxes
 - HierearchyLevel should be set to 1 when the contours of the entomological box are fully (continuously) taken into account by the HSV mask.
2. Once the parameters have been calibrated, click on the button "Freeze parameters"
3. You can deletete wrong detections by listing their identifiers (separated with ",") in de field and clicking on "Delete".
4. a) You can add detections by drawing bounding boxes on each insect :
 - Click on the "Add detections (ROI)" button to launch the selection window (ROI)
 - Select the ROI with your mouse
 - Press Enter or the space bar to validate the ROI
 - Repeat step 1 and 2 for all detections you want
 - Do not forget to press ESC to save all detections
4. b) If the insects have similar sizes, you can go faster by clicking on the insect to add a detection
 - You need to do point 4. a) for a single insect to define the detection size which will apply to other insects
 - Then, click on the "Add detections (click)" button to launch the selection window (click)
 - Click on the center of each insect with the dimension defined above
 - Do not forget to press ESC to save all detections
 - Note that if there are several groups of insects of the same size, you can repeat the process, each time defining a new size.

It's important to know that adding and deleting detection can be done in any order and as many times as desired.
An image showing the detections, their respective IDs and the total number of detections is generated after each detection is added or deleted, so you can see where you stand.

5. Once all the insects have been annotated correctly, click on the "save all" button to generate the txt annotation file, with each line included :
- the object class (0 = insect in this case)
- normalized coordinates x and y
- normalized dimensions width and height
### Insect Detection Algorithms
Two algorithms using different approaches were used to carry out the detection task : YOLO (You Only Look Once) is a fast object detection method that processes the image in a single pass with a single convolutional network, while Faster R-CNN (Region-based Convolutional Neural Networks) uses a two-pass approach. stages with a region proposal network followed by a classification network for more precise but slower detection.
#### YOLO
To use YOLO you first need to install the package called "Ultralytics". 
You can do that using this command line : 
`pip install ultralytics`

1. TRAIN : to train a pretrained model on a custom dataset, simply run this command line :  
`yolo task=detect mode=train model=yolov8x.pt data=custom_data.yaml epochs=100 imgsz=2000 plots=True device=0,1,2,3 close_mosaic=100`
This line was used to train the attached models but several values can be changed depending on your usage :
 - model= : you can use different model size (see : https://docs.ultralytics.com/models/yolov8/#performance-metrics)
 - imgsz= : this is the most restrictive argument, it corresponds to the size in pixels of the images input to the model,
the higher it is, the more precise the detection will be but will require significant resources. Using the Lucia supercomputer the value of 2000 could be reached, with a standard computer this value must be set much lower otherwise you will get an out of memory error.
 - device= : depend on your configuration. Here we use 4 GPU's
 - close_mosaic= : Disables data augmentation over the last N epochs to stabilize training before it is completed. Here, 100 epochs - 100 close_mosaic epochs = 0 epochs of data augmentation because all the photos of the entomological boxes have the same layout and location so no need to augment the data.

About custom_data.yaml : this file is used to specify details about the custom dataset used for model training. It contains : 
 - Class number and names : Here, two models have 1 class "insect" (coleo.pt and lepido.pt) and the other has 2 classes, "coleo" and "lepido" (coleo_lepido.pt)
 - Image Paths : Paths to the images of the train and validation dataset, it must take the following form :
  
   data
     
   ├─ train  
   │&emsp; ├─ images  
   │&emsp; │ &ensp; ├─ name1.JPG  
   │&emsp; │ &ensp; ├─ name2.JPG  
   │&emsp; │ &ensp; └─ ...  
   │&emsp; └─ labels  
   │&emsp;&emsp;&emsp; ├─ name1.txt  
   │&emsp;&emsp;&emsp; ├─ name2.txt  
   │&emsp;&emsp;&emsp; └─ ...  
   └─ val  
    &emsp;&emsp;   ├─ images  
    &emsp;&emsp;   │&ensp;    ├─ name1.JPG  
    &emsp;&emsp;   │&ensp;    ├─ name2.JPG  
    &emsp;&emsp;   │&ensp;    └─ ...  
    &emsp;&emsp;   └─ labels  
    &emsp;&emsp;&emsp;&emsp;        ├─ name1.txt  
    &emsp;&emsp;&emsp;&emsp;        ├─ name2.txt  
     &emsp;&emsp;&emsp;&emsp;       └─ ...  
 
the output model will be in `runs/detect/train_x_/weights/best.pt`
 
 2. Test (Visualization) : if you want to visualize the result of running YOLO on non-view images, run this command
 `yolo task=detect mode=predict model=model.pt conf=0.5 source=folder save=True imgsz=2000 iou=0 max_det=1000 show_label=False show_conf=False line_width=5`  
 - replace model.pt by the model you want to test
 - replace folder by the name of the folder that contains non-view images
 - try different confidence rates to see which is the most appropriate for the future
 - /!\ you must use the same imsize value as that which was used for training (for attached models imgsz=2000)
 - iou is the overlap rate allowed between 2 objects, 0 is for no overlap. You must adapt this value depending on the overlap present on the entomological boxes
 - you can also adapt the maximum number of detections allowed, by default, it is 300, and since many boxes contain more than 300 this value was set to 1000.
 - adapt the last 3 visualization parameters to your convenience

the output images will be in `runs/detect/predict_x_/`
 
 3. Test (CSV) : if you want to save the number of insects detected as well as several information relating to the size of the detections in a CSV file, the process is a bit different. Run the command  
 `python yolo_csv.py <model.pt> <test folder> <conf> <iou> <out.csv>`  
as for the point above you must specify the model, the test folder containing the test images, the confidence rate and the overlap rate. This time if you will have to specify in addition to that the name of the output csv file
 
#### Faster R-CNN
To use faster r cnn launch the following python scripts

1. Train : run  
   `python frcnn_train.py <path to image folder> <path to label folder> <model.pth>`  
 - the images and labels must be contained in the same order for both folders. A simple solution is to keep the same name, and only have the extension which differs e.g : name1.JPG and name1.txt, as for YOLO.
 - specify the name of the .pth model you want to create

2. Test (Vizualisation) : run  
   `python frcnn_test.py <model.pth> <path to test folder> <conf> <iou> <path to out folder>`  
 - as for YOLO you have to specify the model, the test folder containing the test images, the confidence rate and the overlap rate. You must also provide the path to the folder that will contain all the resulting images.
 
3. Test (CSV) : run  
   `python frcnn_csv.py <model.pth> <path to test folder> <conf> <iou> <path to out csv>`
 - you must specify an output csv file rather than a folder compared to the previous point


### Insect Classification Algorithm

To use the classification algorithm, launch the following python scripts

1. Train :  `python classifier.py`
 - All the parameters (images folder, path to csv files,...) can be changed directly in the python files
2. Test : `python predict.py`
 - Same thing for the parameters 