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

## Utilisation
### Image Stacking Tool

Lauch the GUI using this command : python annotation.py

1. Choose the image that you want to annotate
Two windows will appear to visualize parameter calibration in order to automatically annotate as many insects as possible :
 - the minimum and maximum values of the HSV space (Hue,Saturation,Value) can be selected using the first 6 cursors
 - the following three sliders influence the contours of the positive zones of the HSV mask
 - the minimum and maximum area of the interior of a contour can be defined to be considered a detection using the following 2 cursus
 - the following four cursuers are used to adjust the size and position of bounding boxes
 - HierearchyLevel should be set to 1 when the contours of the entomological box are fully (continuously) taken into account by the HSV mask.
 



