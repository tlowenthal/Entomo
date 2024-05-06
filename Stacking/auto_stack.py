
#import
import os
import sys
import subprocess
import time

#function to stack all boxes that are in the directory
def auto_stack(path,n_pic_zone):
    if os.path.isdir(path):
        count_dir = 0
        for box in os.listdir(path):
            path_box = os.path.join(path, box)
            if os.path.isdir(path_box):
                count_dir += 1
        curr_dir = 1
        t_i = time.time()
        for box in os.listdir(path):
            path_box = os.path.join(path, box)
            if os.path.isdir(path_box):
                print('BOX ['+str(curr_dir)+'/'+str(count_dir)+'] :',box)
                subprocess.run(["python", "stacking.py", path_box, n_pic_zone])
          
                curr_dir += 1
        t_f = time.time()
        print("TOTAL_TIME : ",round(t_f-t_i,1), "s")
                
                
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python auto_stack.py <path_to_boxes> <number_pic_per_zone>")
        sys.exit(1)
        
    path = sys.argv[1]
    n_pic_zone = sys.argv[2]
    
    auto_stack(path,n_pic_zone)
