import numpy as np
import json
import cv2
import os
import tqdm
import glob
def is_file_empty(file_path):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

def data_prepare(annotations_file="data/annotations/instances_train2017.json",train_folder="data/train2017"):
    """ Load the annotations file and create the bounding box files for the respective images"""
    # Open Annotations file and change the given image annotations into proper format
    file=open(annotations_file,'r')
    data=json.load(file)
    images={}
    key=["filename","height","width","bbox"]

    for i in data["images"]:
        images[i["id"]]=[i["file_name"].split(".")[0],i["height"],i["width"]]
    for i in data["annotations"]:
        i['bbox'][0]/=images[i['image_id']][2]
        i['bbox'][2]/=images[i['image_id']][2]
        i['bbox'][1]/=images[i['image_id']][1]
        i['bbox'][3]/=images[i['image_id']][1]
        i['bbox'].insert(0,str(i['category_id']-1))
        images[i['image_id']].append(i['bbox'])

    folder=train_folder+"/"
    for i in tqdm.tqdm(images):
        txt=open(os.path.join(folder,images[i][0]+".txt"),'w')
        for j in images[i][3:]:
            j[1]/=2
            j[2]/=2
            temp=map(str,j)
            txt.write(" ".join(temp)+"\n")
        
    text_files=glob.glob(train_folder+"/*.txt")
    temp=0
    for i in tqdm.tqdm(text_files):
        if is_file_empty(i):
            os.remove(i)
            img=train_folder+"/"+os.path.basename(i).rsplit(".")[0]
            img+=".jpg"
            os.remove(img)
            print(i,img)
            temp+=1
    text_files=glob.glob(train_folder+"/*.txt")
    jpg_files=glob.glob(train_folder+"/*.jpg")
    assert len(text_files)==len(jpg_files),"Image and Text file number mismatch"

data_prepare()