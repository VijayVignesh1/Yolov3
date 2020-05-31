from model import *
import torch
from util import DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
net_info,modules=build_net('cfg/yolov3.cfg')
# print(b[4][0].skip)
# print(len(b))
        
# for i in range(106):
#     if "Empty" in str(b[i][0]):
#         print(i)
#         exit(0)
class DarkNet(torch.nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        # self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = build_net('cfg/yolov3.cfg')
        self.cuda="cuda"
    def forward(self,x,targets=None):
        outputs={}
        write=0
        # detections=""
        # global detections
        # detect=[]
        
        for i in range(len(self.module_list)):
            # print(self.module_list[i])
            if "conv" in str(self.module_list[i]) or "up" in str(self.module_list[i]):
                x=self.module_list[i](x)
            elif "route" in str(self.module_list[i]):
                assert self.module_list[i][0].is_route==True
                start=self.module_list[i][0].start
                end=self.module_list[i][0].end
                # print("asa",start)
                if end==0:
                    x=outputs[i+(start)]
                    # print(x.shape)
                else:
                    map1=outputs[i+(start)]
                    map2=outputs[i+(end)]
                    x=torch.cat((map1,map2),1)
            elif "shortcut" in str(self.module_list[i]):
                assert self.module_list[i][0].is_shortcut==True
                skip=self.module_list[i][0].skip
                # print(outputs[i-1].shape,outputs[i+skip+1].shape)
                # print(skip,i)
                x=outputs[i-1]+outputs[i+(skip)]
            elif "detection" in str(self.module_list[i]):
                anchors=self.module_list[i][0].anchors
                inp_dim=int(self.net_info["height"])
                num_classes=80
                x=x.data
                data=self.module_list[i][0](x,inp_dim,anchors,num_classes,True,targets)
                # print("shape",len(x))
                # exit(0)
                # x=predict_transform(x,inp_dim,anchors,num_classes,True)
                x=data[0]
                losses=data[1:]
                print("Loss",losses)
                if not write:
                    # global detections
                    detections=x
                    write=1
                    # detect.append(detections)
                    # print("deee",detect)
                else:
                    # global detections
                    # print("xxxxxxxxxxxx",data[1])
                    detections=torch.cat((detections,x),1)
                    # detect[0]=detections
            outputs[i]=x
            # print(i,x.shape)
        # global detections
        return detections

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

model = DarkNet()
data=DataLoader(416,"dog-cycle-car")
dataloader=torch.utils.data.DataLoader(dataset=data,batch_size=1,num_workers=0)
# print(iter(dataloader).next())
_,target=iter(dataloader).next()
# exit(0)
# print(target)
model=model.to("cuda")
inp = get_test_input()
pred = model(inp.cuda(),target.cuda())
print (pred.shape)