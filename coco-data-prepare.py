import numpy as np
import json
import cv2
import os
import tqdm
# img=cv2.imread("data/train2017/000000391895.jpg")
# x=359/640
# y=146/360
# w=112/640
# h=213/360
# img=cv2.resize(img,(500,700))
# print(x,y,w,h)
# img=cv2.rectangle(img,(int(x*500),int(y*700)),(int((x+w)*500),int((y+h)*700)),(0,255,0),2)
# # img=cv2.rectangle(img,(339,30),(339+153,30+300),(0,255,0),2)
# # img=cv2.rectangle(img,(471,172),(471+35,172+48),(0,255,0),2)
# cv2.imshow('',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit(0)
file=open("data/annotations/instances_train2017.json",'r')
data=json.load(file)
# print(len(data["images"]))
# print(data["annotations"][0])
# print(data["images"][0])
images={}
key=["filename","height","width","bbox"]

for i in data["images"]:
    images[i["id"]]=[i["file_name"].split(".")[0],i["height"],i["width"]]
    # print(i["id"])
    # print(i["file_name"].split(".")[0])
for i in data["annotations"]:
    i['bbox'][0]/=images[i['image_id']][2]
    i['bbox'][2]/=images[i['image_id']][2]
    i['bbox'][1]/=images[i['image_id']][1]
    i['bbox'][3]/=images[i['image_id']][1]
    i['bbox'].insert(0,str(i['category_id']-1))
    images[i['image_id']].append(i['bbox'])

# print(len(images.keys()))
# print(images[391895])
folder="data/train2017/"
for i in tqdm.tqdm(images):
    txt=open(os.path.join(folder,images[i][0]+".txt"),'w')
    for j in images[i][3:]:
        j[1]/=2
        j[2]/=2
        temp=map(str,j)
        txt.write(" ".join(temp)+"\n")
    
"""
print(images[391895])
temp=images[391895]
img=cv2.resize(img,(416,416))
for i in temp[3:]:
    img=cv2.rectangle(img,(int(i[0]*416),int(i[1]*416)),(int(i[0]*416)+(int(i[2]*416)),int(i[1]*416)+(int(i[3]*416))),(0,255,0),2)
cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
