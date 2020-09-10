import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import cv2
import os
class DataLoader(Dataset):
    """ Loads the data into proper format for training the model """

    def __init__(self,img_size,train_path,is_train=True):
        self.img_size=img_size
        self.max_objects=50
        self.is_train=is_train
        self.train_path=train_path
        self.images=glob.glob(self.train_path+"/*.jpg")

        # Uncomment below to train on a small subset of the dataset. 
        # self.images=self.images[:1000]

        self.total_train=len(self.images)

    def __getitem__(self,index):
        # Read image and resize
        img=cv2.imread(self.images[index])
        img=cv2.resize(img,(self.img_size,self.img_size))
        oldX,oldY,_=img.shape
        Rx,Ry=self.img_size/oldX,self.img_size/oldY

        # Read the text file for the given image
        points_path=os.path.basename(self.images[index]).rsplit(".")[0]
        points_path+=".txt"
        points_path=os.path.join(self.train_path,points_path)
        points=np.loadtxt(points_path)

        # Convert image into FloatTensor and Normalize
        tensor_image=torch.FloatTensor(img)
        tensor_image=tensor_image.permute(2,0,1)
        tensor_image/=255.

        # Check if the text file has annotations and return them as tensors
        assert len(points)!=0
        filled_labels=np.zeros((self.max_objects,5))
        filled_labels[range(len(points))[:self.max_objects]]=points[:self.max_objects]
        filled_labels=torch.from_numpy(filled_labels)
        return tensor_image,filled_labels

    def __len__(self):
        return self.total_train