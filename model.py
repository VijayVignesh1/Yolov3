import torch
from util import *

def build_net(cfg):
    data=open(cfg,'r')
    count=0
    lines=data.readlines()
    # print(len(lines))
    # print(lines)
    # exit(0)
    layer=0
    modules=torch.nn.ModuleList()
    output_filters=[3]
    for i in range(len(lines)):
        # print(i)
        # print(lines[i])


        if lines[i][1:-2]=="convolutional":
            # print(lines[i])
            # print("CONV")
            i+=1
            temp={}
            seq=torch.nn.Sequential()
            while lines[i]!="\n":
                # print(lines[i].split("=")[1].rstrip("\n"))
                key,value=lines[i].split("=")[0],lines[i].split("=")[1].rstrip("\n")
                key,value=key.strip(),value.strip()
                temp[key]=value
                # print(lines[i])
                i+=1
            # print(temp)
            if int(temp['pad']):
                pad=(int(temp["size"])-1)//2
            else:
                pad=0
            conv=torch.nn.Conv2d(output_filters[-1],int(temp["filters"]),int(temp["size"]),int(temp["stride"]),pad)
            seq.add_module("conv_%d"%(layer),conv)
            try:
                if temp["batch_normalize"]:
                    norm=torch.nn.BatchNorm2d(int(temp['filters']))
                    seq.add_module("norm_%d"%(layer),norm)
            except:
                pass
            if temp['activation']=="leaky":
                act=torch.nn.LeakyReLU(0.1,inplace=True)
                seq.add_module("act_%d"%(layer),act)
            # seq=torch.nn.Sequential(conv,norm,act)
            layer+=1
            # print(modules)
            output_filters.append(int(temp['filters']))
            modules.append(seq)

        elif lines[i][1:-2]=="upsample":
            i+=1
            temp={}
            seq=torch.nn.Sequential()
            while lines[i]!="\n":
                key,value=lines[i].split("=")[0],lines[i].split("=")[1].rstrip("\n")
                key,value=key.strip(),value.strip()
                temp[key]=value
                i+=1            
            up=torch.nn.Upsample(scale_factor=2,mode="bilinear")
            seq.add_module("up_%d"%(layer),up)
            output_filters.append(output_filters[-1])
            modules.append(seq)
            layer+=1

        elif lines[i][1:-2]=="route":
            i+=1
            temp={}
            seq=torch.nn.Sequential()
            while lines[i]!='\n':
                key,value=lines[i].split("=")[0],lines[i].split("=")[1].rstrip("\n")
                key,value=key.strip(),value.strip()
                temp[key]=value
                i+=1
            if "," in temp['layers']:
                start,end=temp['layers'].split(',')
            else:
                start=temp['layers']
                end=0
            start=int(start)
            end=int(end)    
            # print(start,end)  
            if start>0:
                start-=layer
            if end>0:
                end-=layer
            route=EmptyLayer(is_route=True,start=start,end=end)
            seq.add_module("route_%d"%(layer),route)
            # print(start,end,layer+start)
            # print(len(output_filters))
            filters=output_filters[layer + start + 1]
            # print(layer,filters,start,end)
            if end<0:
                # assert output_filters[-1]==output_filters[skip]
                # print(filters,output_filters[layer + end])
                filters+=output_filters[layer + end + 1]
            output_filters.append(filters)
            modules.append(seq)
            layer+=1

        elif lines[i][1:-2]=='shortcut':
            i+=1
            temp={}
            seq=torch.nn.Sequential()
            while lines[i]!="\n":
                key,value=lines[i].split("=")[0],lines[i].split("=")[1].rstrip("\n")
                key,value=key.strip(),value.strip()
                temp[key]=value
                i+=1
            # shortcut=EmptyLayer()
            skip=int(temp['from'])
            if skip>0:
                skip-=layer
            assert output_filters[-1]==output_filters[skip]
            output_filters.append(output_filters[-1])  
            shortcut=EmptyLayer(is_shortcut=True,skip=skip)    
            seq.add_module("shortcut_%d"%(layer),shortcut)
            modules.append(seq)
            layer+=1

        elif lines[i][1:-2]=="yolo":
            # print(lines[i])
            i+=1
            temp={}
            seq=torch.nn.Sequential()
            while lines[i]!="\n":
                key,value=lines[i].split("=")[0],lines[i].split("=")[1].rstrip("\n")
                key,value=key.strip(),value.strip()
                temp[key]=value
                i+=1
            # print(temp)
            masks=temp["mask"].split(',')
            masks=[int(i) for i in masks]
            anchors=temp['anchors'].split(",")
            anchors=[(int(anchors[i]),int(anchors[i+1])) for i in range(0,len(anchors),2)]
            anchors=[anchors[i] for i in masks]
            detection=DetectionLayer(anchors)
            seq.add_module("detection_%d"%(layer),detection)
            output_filters.append(output_filters[-1])
            modules.append(seq)
            layer+=1

        elif lines[i][1:-2]=="net":
            # print("NET")
            i+=1
            net_info={}
            while lines[i]!="\n":
                if lines[i][0]=="#":
                    i+=1
                    continue
                # print(lines[i])
                key,value=lines[i].split("=")[0],lines[i].split("=")[1].rstrip("\n")
                key,value=key.strip(),value.strip()
                try:
                    if "." in value:
                        net_info[key]=float(value)
                    else:
                        net_info[key]=int(value)
                except:
                    pass
                i+=1    
        # modules.append(seq)
        # layer+=1

        # if count==25:
        #     break
        # count+=1
    # print(modules)
    # print(output_filters)
    # print("ad",output_filters[:84])
    # print(net_info)
    return net_info,modules