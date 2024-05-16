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

## Use
### Image Stacking Tool

To stack photos of several boxes, you simply has to run this command : python auto_stack.py 'path to the folder' 'number of pictures per zone' where :
 - 'path to the folder' points to the folder containing all the boxes to be stacked. it must take the following form :
  
Folder  
   ├─ Box 1  
   │  ├─ Image 1  
   │  ├─ Image 2  
   │  └─ ...  
   ├─ Box 2  
   │  ├─ Image 1  
   │  ├─ Image 2  
   │  └─ ...  
   └─ ...  
   
 - 'number of pictures per zone' is the number of picture of the same zone taken at differents focus

### Image Annotation Tool
Lauch the GUI using this command : python annotation.py

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
