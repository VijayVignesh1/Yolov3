from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt

def showTensor(aTensor):
    """ Plots the given tensor in matplotlib """
    plt.figure()
    plt.imshow(aTensor.numpy())
    plt.colorbar()
    plt.show()

class EmptyLayer(torch.nn.Module):
    """ An Empty Layer used for route and shortcut layers which just uses the previous layers """
    def __init__(self, is_route=False, is_shortcut=False,start=1,end=1,skip=1):
        super(EmptyLayer, self).__init__()
        self.is_route=is_route
        self.is_shortcut=is_shortcut
        self.start=start
        self.end=end
        self.skip=skip

class DetectionLayer(torch.nn.Module):
    """ Detection Layer used to predict the bounding boxes """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
        self.ce_loss = nn.CrossEntropyLoss() 
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self,prediction,inp_dim,anchors,num_classes,CUDA=True,targets=None):

        # Reshape the the input to output 85 columns (0-3 for bounding box coordinates, 4th for objectness and 5-85 for class prediction)
        batch_size=prediction.size(0)
        stride=inp_dim//prediction.size(2)
        grid_size=inp_dim//stride
        bbox_attrs=5+num_classes
        num_anchors=len(anchors)
        prediction=prediction.view(batch_size,num_anchors,bbox_attrs,grid_size,grid_size).permute(0,1,3,4,2).contiguous()
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

        # Apply Sigmoid to the predictions
        prediction[...,0]=torch.sigmoid(prediction[...,0])
        prediction[...,1]=torch.sigmoid(prediction[...,1])
        prediction[...,4]=torch.sigmoid(prediction[...,4])
        prediction[...,5:]=torch.sigmoid(prediction[...,5:])
        x,y,w,h,pred_conf,pred_cls=prediction[...,0].clone(),prediction[...,1].clone(),prediction[...,2].clone(),prediction[...,3].clone(),prediction[...,4],prediction[...,5:]
        
        # Add the x and y offsets to the predictions
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(1,1,grid_size,grid_size)
        y_offset = torch.FloatTensor(b).view(1,1,grid_size,grid_size)
        if CUDA:
            x_offset=x_offset.cuda()
            y_offset=y_offset.cuda()
        prediction[...,0]=x.detach()+x_offset
        prediction[...,1]=y.detach()+y_offset
        
        # Multiply anchor to the width and height predictions
        anchors = torch.FloatTensor(anchors)
        anchor_w=anchors[:,0:1].view((1,num_anchors,1,1))
        anchor_h=anchors[:,1:2].view((1,num_anchors,1,1))
        if CUDA:
            anchor_h=anchor_h.cuda()
            anchor_w=anchor_w.cuda()
        prediction[...,2] = torch.exp(w.detach())*anchor_w
        prediction[...,3] = torch.exp(h.detach())*anchor_h
        
        pred=torch.cat((prediction[...,:4].view(batch_size,-1,4) * stride,
                        prediction[...,4].view(batch_size,-1,1),
                        prediction[...,5:].view(batch_size,-1,num_classes)),-1)
        pred=pred.view(batch_size,num_anchors,grid_size,grid_size,bbox_attrs).permute(0,2,3,1,4).contiguous()
        pred=pred.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)

        # If test, return the prediction, else calculate the masks for training
        if targets is None:
            return pred

        # Prepare the ground truth values in proper format for training
        num_obj,num_correct,mask,conf_mask,tx,ty,tw,th,tconf,tcls=TargetsReady(prediction[...,:4].cpu().data,prediction[...,4].cpu().data,prediction[...,5:],targets.cpu().data,anchors,num_anchors,
                    num_classes,grid_size,0.5,416)

        tcls=tcls.type(torch.LongTensor)
        num_prop=int((pred_conf>0.5).sum().item())
        recall=float(num_correct/num_obj) if num_obj else 1
        precision=float(num_correct/num_prop)

        conf_mask_true=mask.type(torch.ByteTensor)
        conf_mask_false=(conf_mask-mask).type(torch.ByteTensor)
        tconf=tconf.float()
        mask=mask.type(torch.ByteTensor)
    
        tx = Variable(tx, requires_grad=False)
        ty = Variable(ty, requires_grad=False)
        tw = Variable(tw, requires_grad=False)
        th = Variable(th, requires_grad=False)
        tconf = Variable(tconf, requires_grad=False)
        tcls = Variable(tcls, requires_grad=False)

        if CUDA:
            tx=tx.cuda()
            ty=ty.cuda()
            tw=tw.cuda()
            th=th.cuda()
            tconf=tconf.cuda()
            tcls=tcls.cuda()
            self.mse_loss=self.mse_loss.cuda()
            self.ce_loss=self.ce_loss.cuda()
                
        # Calculate the loss using the predicted values and the prepared ground truth values and return them
        loss_x=(self.lambda_coord*self.mse_loss(x[mask],tx[mask]))/batch_size
        loss_y=(self.lambda_coord*self.mse_loss(y[mask],ty[mask]))/batch_size
        loss_w=(self.lambda_coord*self.mse_loss(w[mask],tw[mask]))/batch_size
        loss_h=(self.lambda_coord*self.mse_loss(h[mask],th[mask]))/batch_size

        loss_conf=(self.lambda_noobj*(self.mse_loss(pred_conf[conf_mask_false],tconf[conf_mask_false])
                                    +self.mse_loss(pred_conf[conf_mask_true],tconf[conf_mask_true])))/batch_size
        loss_cls=(1/batch_size)*(self.ce_loss(pred_cls[mask],torch.argmax(tcls[mask],1)))

        loss=loss_x+loss_y+loss_w+loss_h+loss_conf+loss_cls
        recall/=3
        precision/=3
        return [loss,loss_x.item(),loss_y.item(),loss_w.item(),loss_h.item(),loss_conf.item(),loss_cls.item(),recall,precision]
        
def TargetsReady(pred_coord,pred_conf,pred_class,targets,anchors,num_anchors,num_classes,grid_size,ignore_thresh,img_size):
    """ Prepare the targets/Ground Truth values for training in a proper format """

    batch_size=targets.size(0)
    mask=torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    conf_mask=torch.ones(batch_size,num_anchors,grid_size,grid_size)
    tx=torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    ty=torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tw=torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    th=torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tconf=torch.zeros(batch_size,num_anchors,grid_size,grid_size).type(torch.ByteTensor)
    tcls=torch.zeros(batch_size,num_anchors,grid_size,grid_size,num_classes).type(torch.ByteTensor)

    num_gt=0
    num_correct=0
    for batch in range(batch_size):
        for target in range(targets.shape[1]):
            if targets[batch,target,:].sum()==0:
                continue
            num_gt+=1
            

            gx=targets[batch,target,1]*grid_size
            gy=targets[batch,target,2]*grid_size
            gw=targets[batch,target,3]*grid_size
            gh=targets[batch,target,4]*grid_size

            gi=gx.int()
            gj=gy.int()

            # Assert if prediction is less than the grid size
            assert gi<=grid_size and gj<=grid_size

            gt_box=torch.FloatTensor([0,0,gw,gh]).unsqueeze(0)
            anchor_box=torch.FloatTensor(np.concatenate((np.zeros((len(anchors),2)),np.array(anchors)),1))
            anchor_iou=IOU(gt_box,anchor_box) 

            # Create the mask,confidence mask, target x, y,w and h and return them
            conf_mask[batch,anchor_iou>ignore_thresh,gj,gi]=0
            best=np.argmax(anchor_iou)

            gt_box=torch.FloatTensor([gx,gy,gw,gh]).unsqueeze(0)
            pred_box=pred_coord[batch,best,gj,gi].type(torch.FloatTensor).unsqueeze(0)

            mask[batch,best,gj,gi]=1
            conf_mask[batch,best,gj,gi]=1

            tx[batch,best,gj,gi]=gx-gi
            ty[batch,best,gj,gi]=gy-gj

            tw[batch,best,gj,gi]=math.log(gw / anchors[best][0] + 1e-16)
            th[batch,best,gj,gi]=math.log(gh / anchors[best][1] + 1e-16)

            label=int(targets[batch,target,0])
            tcls[batch,best,gj,gi,label]=1
            tconf[batch,best,gj,gi]=1

            iou=IOU(gt_box,pred_box,True)


            pred_label=torch.argmax(pred_class[batch,best,gj,gi])
            score=pred_conf[batch,best,gj,gi]

            if iou>0.5 and pred_label==label and score>0.5:
                num_correct+=1
    return num_gt,num_correct,mask,conf_mask,tx,ty,tw,th,tconf,tcls


def IOU(box1,box2,center=False):
    """ Calculates the Intersection Over Union of the given two boxes """
    if center:
        b1_x1,b1_x2 = box1[:,0]-box1[:,2]/2,box1[:,0]+box1[:,2]/2
        b2_x1,b2_x2 = box2[:,0]-box2[:,2]/2,box2[:,0]+box2[:,2]/2
        b1_y1,b1_y2 = box1[:,1]-box1[:,3]/2,box1[:,1]+box1[:,3]/2
        b2_y1,b2_y2 = box2[:,1]-box2[:,3]/2,box2[:,1]+box2[:,3]/2
    else:
        b1_x1,b1_x2,b1_y1,b1_y2=box1[:,0],box1[:,2],box1[:,1],box1[:,3]
        b2_x1,b2_x2,b2_y1,b2_y2=box2[:,0],box2[:,2],box2[:,1],box2[:,3]

    i_x1=torch.max(b1_x1,b2_x1)
    i_y1=torch.max(b1_y1,b2_y1)
    i_x2=torch.min(b1_x2,b2_x2)
    i_y2=torch.min(b1_y2,b2_y2)

    zero=torch.zeros((i_x2.shape))
    area=torch.max(zero,i_x2-i_x1+1)*torch.max(zero,i_y2-i_y1+1)

    box1Area=(b1_x2-b1_x1+1)*(b1_y2-b1_y1+1)
    box2Area=(b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)
        
    iou=area/(box1Area+box2Area-area+1e-16)

    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    """ Reshape the prediction of the network into proper shape for finding the bounding boxes """

    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    print(prediction[:,:,:2])
    prediction[:,:,:2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride

    return prediction