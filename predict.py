import torch
from yolo import DarkNet
import cv2
from util import IOU
import numpy as np

def nonMaxSuppression(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    """ Performs Non Maximal Suppression on the given prediction """

    # Ignore the prediction with values less than treshold
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
        # For every predicted class ignore the box of low prediction confidence if two boxes of same class overlap more than threshold
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

            # Sort the boxes of the same class, helps in deciding which box to drop in case of overlap
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
    """ Return the unique rows of the given data """
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])

def predict(img, prediction, conf_thresh, num_classes):
    """ Predicts the bounding box using the given prediction"""

    # Bring the given prediction in proper format
    prediction=prediction.data.cpu()
    conf=(prediction[:,:,4]>conf_thresh).float().unsqueeze(2)
    prediction*=conf
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    for i in range(prediction.shape[0]):

        # Pluck out all the non zeros predictions and find the class it belongs to
        nonzero_index=prediction[i,:,4].data.cpu().nonzero()
        final_pred=prediction[i,nonzero_index.squeeze(1)]
        classes=final_pred[:,5:]
        scores,classes=torch.max(classes,dim=1)
        scores=scores.data
        classes=classes.data
        classes=classes.detach().cpu().numpy()
        dictionary={}
        index=torch.sort(final_pred[:,4],descending=True)[1]
        final_pred=final_pred[index]

        # Find the list of coordinates for every class
        for j in range(final_pred.shape[0]):
            temp_classes=final_pred[j,5:]
            temp_score,temp_cls=torch.max(temp_classes.unsqueeze(0),1)
            temp_cls=str(temp_cls.data.cpu().numpy()[0])
            temp_coord=final_pred[j,:4].data.cpu().numpy()
            if temp_cls in dictionary:
                dictionary[temp_cls].append(list(temp_coord))
            else:
                dictionary[temp_cls]=[list(temp_coord)]

        # For every class find if any two bounding boxes are overlapping more than the threshold using the IOU and drop the box with 
        # low confidence
        for k in dictionary:
            count=0
            while count+2!=len(dictionary[k]):
                compare_tens=torch.Tensor(dictionary[k][count+1:])
                if compare_tens.shape==torch.Size([0]):
                    break

                ious=IOU(torch.Tensor(dictionary[k][count]).unsqueeze(0),torch.Tensor(dictionary[k][count+1:]))
                ious=ious.unsqueeze(1)
                ious=ious.numpy()
                conf=(ious<0.02)

                tmp=[list(compare_tens[i].numpy()) for i in range(len(conf)) if conf[i]==True]
                dictionary[k]=dictionary[k][:count+1]+tmp
                count+=1  

        # Draw the final set of unique rectangles onto the images
        print("Class\tX\tY\tX1\tY2")
        for k in dictionary:
            dictionary[k]= unique_rows(np.array(dictionary[k]))
            for x,y,x1,y1 in dictionary[k]:
                print("%s\t%.1f\t%.1f\t%.1f\t%.1f\n"%(k,float(x),float(y),float(x1),float(y1)))
                img=cv2.rectangle(img,(x,y),(x1,y1),(0,255,0),2)

    # Display and Save the Image
    cv2.imshow('Detection',img)
    cv2.imwrite('images/Result.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """ Test the model on test images """
    checkpoint_file='checkpoints/checkpoint.epoch.2.6999.pth.tar'
    image_file='data/train/Buffy_105.jpg'
    img=cv2.imread(image_file)
    img=cv2.resize(img,(416,416))
    img1=img.copy()
    img=torch.from_numpy(img)
    img=img.unsqueeze(0).permute(0,3,1,2)
    img/=255.
    img=img.type(torch.FloatTensor)
    model=DarkNet()
    model=model.to("cuda")
    model.load_state_dict(torch.load(checkpoint_file)['state_dict'])
    prediction=model(img.cuda(),None)
    predict(img1,prediction,0.7,1)

if __name__=="__main__":
    main()
