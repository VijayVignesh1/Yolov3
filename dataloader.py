import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import cv2
import os
"""
class DataLoader(Dataset):
    def __init__(self,img_size,train_path,is_train=True):
        # print("hi")
        self.img_size=img_size
        self.img_shape=(self.img_size,self.img_size)
        self.max_objects=50
        self.is_train=is_train
        self.train_path=train_path
    def __getitem__(self,index):
        img=cv2.imread(self.train_path+".png")
        # print(img)
        img=cv2.resize(img,self.img_shape)
        img=np.transpose(img,(2,0,1))
        img=torch.from_numpy(img).float()

        labels=np.loadtxt(self.train_path+".txt")
        # labels[:,1]*=self.img_size
        # labels[:,2]*=self.img_size
        # labels[:,3]*=self.img_size
        # labels[:,4]*=self.img_size
        filled_labels=np.zeros((self.max_objects,5))
        filled_labels[range(len(labels))[:self.max_objects]]=labels[:self.max_objects]
        filled_labels=torch.from_numpy(filled_labels)
        # print(filled_labels)
        return img,filled_labels
    def __len__(self):
        return 1
"""
class DataLoader(Dataset):
    def __init__(self,img_size,train_path,is_train=True):
        self.img_size=img_size
        self.max_objects=50
        self.is_train=is_train
        self.train_path=train_path
        self.images=glob.glob(self.train_path+"/*.jpg")
        # annotation=glob.glob(self.train_path+"/*.txt")
        self.total_train=len(self.images)
    def __getitem__(self,index):
        # print(self.train_path)
        img=cv2.imread(self.images[index])
        img=cv2.resize(img,(self.img_size,self.img_size))
        # print(img.shape)
        oldX,oldY,_=img.shape
        Rx,Ry=self.img_size/oldX,self.img_size/oldY
        points_path=os.path.basename(self.images[index]).rsplit(".")[0]
        points_path+=".txt"
        points_path=os.path.join(self.train_path,points_path)
        # print(points_path)
        points=np.loadtxt(points_path)
        # print(points_path)
        # print(points)
        # try:
        #     x,y=points[:,1]*self.img_size,points[:,2]*self.img_size
        # except IndexError:
        #     points=points.reshape((1,points.shape[0]))
        #     x,y=points[:,1]*self.img_size,points[:,2]*self.img_size
        # print(x[0],y[0])
        # img=cv2.circle(img,(int(x[0]),int(y[0])),2,(0,255,0),-1)
        # cv2.imshow("",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit(0)
        tensor_image=torch.FloatTensor(img)
        tensor_image=tensor_image.permute(2,0,1)
        tensor_image/=255.
        # print(tensor_image.shape)
        filled_labels=np.zeros((self.max_objects,5))
        filled_labels[range(len(points))[:self.max_objects]]=points[:self.max_objects]
        filled_labels=torch.from_numpy(filled_labels)
        # print(filled_labels.shape)
        return tensor_image,filled_labels
    def __len__(self):
        return self.total_train