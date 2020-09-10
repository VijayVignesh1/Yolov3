from model import *
import torch
from dataloader import DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class DarkNet(torch.nn.Module):
    """ The DarkNet model class """
    def __init__(self):
        super(DarkNet, self).__init__()

        # Build the network using the config file
        self.net_info, self.module_list = build_net('cfg/yolov3 - Copy.cfg')
        self.cuda="cuda"
        self.losses=np.array([0,0,0,0,0,0,0,0])
    def forward(self,x,targets=None):
        train_output=[]
        outputs={}
        write=0
        self.losses=np.array([0,0,0,0,0,0,0,0])
        for i in range(len(self.module_list)):

            # If conv, add the conv layer
            if "conv" in str(self.module_list[i]) or "up" in str(self.module_list[i]):
                x=self.module_list[i](x)

            # If route, combine the given previous layers
            elif "route" in str(self.module_list[i]):
                assert self.module_list[i][0].is_route==True
                start=self.module_list[i][0].start
                end=self.module_list[i][0].end
                if end==0:
                    x=outputs[i+(start)]
                else:
                    map1=outputs[i+(start)]
                    map2=outputs[i+(end)]
                    x=torch.cat((map1,map2),1)
            
            # If shortcut, add the skip connections
            elif "shortcut" in str(self.module_list[i]):
                assert self.module_list[i][0].is_shortcut==True
                skip=self.module_list[i][0].skip
                x=outputs[i-1]+outputs[i+(skip)]
            
            # If detection, detect the bounding boxes and return
            elif "detection" in str(self.module_list[i]):
                anchors=self.module_list[i][0].anchors
                inp_dim=int(self.net_info["height"])

                # Change number_classes as per dataset
                # num_classes=80
                num_classes=1
                if targets is not None:
                    data=self.module_list[i][0](x,inp_dim,anchors,num_classes,True,targets)
                    x=data[0]
                    loss=np.array(data[1:])

                    self.losses=np.sum((self.losses,loss),axis=0)
                
                    train_output.append(x)
                else:
                    x=self.module_list[i][0](x,inp_dim,anchors,num_classes,True,None)
                    if not write:
                        detections=x
                        write=1
                    else:
                        detections=torch.cat((detections,x),1)
            outputs[i]=x

        return detections if targets is None else sum(train_output)

def get_test_input():
    """ Prepare the test input for testing """
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


