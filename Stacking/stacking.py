#import
import cv2
import os
import glob
import time
import sys
import numpy as np

#load all bmp images from a folder
def load_images(dir_name, file_name_pattern):
    image_paths = glob.glob(os.path.join(dir_name, file_name_pattern))
    image_paths.sort()
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images

#return im2 aligned with im1
def align(im1, im2):
    #covert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)
    #size of image1
    sz = im1.shape

    #define the motion model
    #warp_mode = cv2.MOTION_HOMOGRAPHY
    #warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = cv2.MOTION_AFFINE
    #warp_mode = cv2.MOTION_EUCLIDEAN
    

    #define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    #number of iterations
    number_of_iterations = 50;

    #threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-2;

    #termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    #ECC algorithm, results are stored in warp_matrix
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None, gaussFiltSize=3)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        #warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), 
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        #warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), 
                                     flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    return im2_aligned    

#align all images
def align_images(images):
    im0 = images[int(len(images)/2)]
    aligned_image_list = []
    count=0
    
    for im in images:
        count+=1
        print('            IMAGE ['+str(count)+'/'+str(len(images))+']', end="\r")
        aligned_image_list.append(align(im0, im))
    return aligned_image_list

#compute the gradient map of the image
def doLap(image):
    kernel_size = 9         
    blur_size = 1          
    blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

#merge best focus
def focus_stack(unimages):
    print('        ALINGNING')
    images = align_images(unimages)
    print('        FOCUS_STACKING    ')
    laps = []
    for i in range(len(images)):
        laps.append(doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)))

    laps = np.asarray(laps)
    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    for y in range(0,images[0].shape[0]):
        for x in range(0, images[0].shape[1]):
            yxlaps = abs(laps[:, y, x])
            index = (np.where(yxlaps == max(yxlaps)))[0][0]
            output[y,x] = images[index][y,x]
    return output

###MAIN###
#folder paths
file_name_pattern = '*.bmp'
dir_name_in = sys.argv[1] # input folder

data_out = "stacked"
if not os.path.exists(data_out):
        os.makedirs(data_out)
if not os.path.exists(os.path.join(data_out,os.path.basename(dir_name_in))):
    #sub-folder with the name of the box
    os.mkdir(os.path.join(data_out,os.path.basename(dir_name_in)))
dir_name_out = os.path.join(data_out,os.path.basename(dir_name_in))
n_picpzone = int(sys.argv[2]) # number of picture per zone

#counting files 
count_files = 0
#iterate throught directory
for path in os.listdir(dir_name_in):
    #check if current path is a file
    if os.path.isfile(os.path.join(dir_name_in, path)):
        count_files += 1

n_zones = count_files/n_picpzone
#print("START")
    
images = load_images(dir_name_in, file_name_pattern)

count=0
total_time_i = time.time()
for i in range(0,count_files,n_picpzone): # every iteration is for one zone
    starting_time = time.time()
    
    print('    PART ['+str(i//n_picpzone+1)+'/'+str(count_files//n_picpzone)+']')
    im = focus_stack(images[i:i+n_picpzone])
    #print('LOADING files in {}'.format(dir_name_in))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    count+=1
    cv2.imwrite(dir_name_out+'/focus_stacked'+str(count)+'.bmp',im)    
    ending_time = time.time()
    print("        TIME : ",round(ending_time-starting_time,1), "s")
total_time_f = time.time()
print("    BOX_TIME : ",round(total_time_f-total_time_i,1), "s")
