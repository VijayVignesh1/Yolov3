import torch
from yolo import DarkNet
import cv2
from util import IOU,bboxIOU
import numpy as np

def nonMaxSuppression(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            if nms:
                for i in range(idx):
                    try:
                        ious = bboxIOU(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:],True)
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output


def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])

def predict(img, prediction, conf_thresh, num_classes):
    # print(prediction)
    prediction=prediction.data.cpu()
    # print(prediction[:5])
    # exit(0)
    # output=nonMaxSuppression(prediction,0.5,91)
    # print(output)
    # exit(0)
    # print(prediction[:,:,4])
    conf=(prediction[:,:,4]>conf_thresh).float().unsqueeze(2)
    prediction*=conf
    # print(prediction[:,:,4].nonzero()[:,1])
    # print(prediction[prediction[:,:,4].nonzero()[:,1]].shape)
    # exit(0)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    # box_corner[box_corner<0]=0
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    # print(box_corner)
    # exit(0)
    for i in range(prediction.shape[0]):
        nonzero_index=prediction[i,:,4].data.cpu().nonzero()
        final_pred=prediction[i,nonzero_index.squeeze(1)]
        # exit(0)
        classes=final_pred[:,5:]
        # print(classes.shape)
        scores,classes=torch.max(classes,dim=1)
        # print(classes)
        scores=scores.data
        classes=classes.data
        classes=classes.detach().cpu().numpy()
        # print(set(classes))
        dictionary={}
        index=torch.sort(final_pred[:,4],descending=True)[1]
        final_pred=final_pred[index]
        # print(final_pred)
        # final_pred=final_pred[final_pred[:,3]>0.0]
        # print(final_pred.shape)
        # exit(0)
        # x,y,x1,y1=final_pred[0][0].numpy(),final_pred[0][1].numpy(),final_pred[0][2].numpy(),final_pred[0][2].numpy()
        # print(x,y,x1,y1)
        # cv2.rectangle(img,(int(x),int(y)),(int(x1),int(y1)),(0,255,0),2)
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit(0)
        for j in range(final_pred.shape[0]):
            temp_classes=final_pred[j,5:]
            temp_score,temp_cls=torch.max(temp_classes.unsqueeze(0),1)
            temp_cls=str(temp_cls.data.cpu().numpy()[0])
            temp_coord=final_pred[j,:4].data.cpu().numpy()
            # print(temp_cls)
            if temp_cls in dictionary:
                dictionary[temp_cls].append(list(temp_coord))
                # print(dictionary[temp_cls])
                # exit(0)
            else:
                dictionary[temp_cls]=[list(temp_coord)]
            # if temp.cls.data.cpu() in dictionary.keys()
            # exit(0)
        # print(dictionary['0'][0])
        # exit(0)
        # print(dictionary['0'][-5:])
        # print(bboxIOU(torch.Tensor([0,0,0,0]).unsqueeze(0),torch.Tensor([0,0,0,0]).unsqueeze(0),True))
        # exit(0)
        
        for k in dictionary:
            count=0
            while count+2!=len(dictionary[k]):
                # try:
                # print(torch.Tensor(dictionary[k][count]).unsqueeze(0).shape,torch.Tensor(dictionary[k][count+1:]).shape)
                # exit(0)
                # print(len(dictionary[k][count+1:]))

                compare_tens=torch.Tensor(dictionary[k][count+1:])
                if compare_tens.shape==torch.Size([0]):
                    break
                # print(compare_tens.shape==torch.Size([0]))
                ious=IOU(torch.Tensor(dictionary[k][count]).unsqueeze(0),torch.Tensor(dictionary[k][count+1:]))
                ious=ious.unsqueeze(1)
                ious=ious.numpy()
                # print("box",torch.Tensor(dictionary[k][count]))
                # print("ious",ious.shape)
                
                # print(ious[:5])
                conf=(ious<0.02)
                # print(conf)
                # exit(0)

                tmp=[list(compare_tens[i].numpy()) for i in range(len(conf)) if conf[i]==True]
                # print(1<0.04)
                # tmp=[]
                # for i in range(len(conf)):
                #     if conf[i]==True:
                #         print(compare_tens[i])
                #         tmp.append(compare_tens[i])

                # print(len(tmp))
                # print("tmp",tmp)
                # exit(0)
                # print(len(dictionary[k][:count+1]))
                # print(dictionary[k][:5])
                dictionary[k]=dictionary[k][:count+1]+tmp
                # print(dictionary[k][:5])
                # dictionary[k][count+1:]=dictionary[k][ious<0.6]
                count+=1  
                # except:
                #     print(len(dictionary[k]))
                #     break
            # print(len(dictionary[k]))
        # exit(0)
        # print(dictionary['0'])

        for k in dictionary:
            dictionary[k]= unique_rows(np.array(dictionary[k]))
            for x,y,x1,y1 in dictionary[k]:
                print(k,x,y,x1,y1)
                img=cv2.rectangle(img,(x,y),(x1,y1),(0,255,0),2)
                # img = cv2.putText(img, k, (int(x)+10,int(y)+10), 1,1,(255,0,0), 2, 1)
    """
    count=0
    print(prediction.shape[1])
    for i in range(prediction.shape[0]):
        if count==1:
            break
        for j in range(1000):
            if prediction[i,j,4]>=conf_thresh:
                x,y=box_corner[i,j,0],box_corner[i,j,1]
                x1,y1=box_corner[i,j,2],box_corner[i,j,3]
                img=cv2.rectangle(img,(x,y),(x1,y1),(0,255,0),2)
        count+=1
        cv2.imshow('',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    cv2.imshow('Detection',img)
    cv2.imwrite('Result.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
img=cv2.imread('data/train/Buffy_105.jpg')
img=cv2.resize(img,(416,416))
img1=img.copy()
img=torch.from_numpy(img)
img=img.unsqueeze(0).permute(0,3,1,2)
img/=255.
img=img.type(torch.FloatTensor)
model=DarkNet()
model=model.to("cuda")
model.load_state_dict(torch.load('checkpoints/checkpoint.epoch.2.6999.pth.tar')['state_dict'])
prediction=model(img.cuda(),None)
# print(prediction.shape)
predict(img1,prediction,0.7,1)
