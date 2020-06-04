from dataloader import DataLoader
import torch
from model import *
from yolo import DarkNet
# checkpoint='checkpoint.epoch.0.pth.tar'
checkpoint=None
start_epoch=0
end_epoch=2
model = DarkNet()
data=DataLoader(416,"data/train2017")
dataloader=torch.utils.data.DataLoader(dataset=data,batch_size=8,num_workers=0,shuffle=True)
# print(iter(dataloader).next())
"""
inp,target=iter(dataloader).next()
# print(inp.shape)
# exit(0)
# print(target)
model=model.to("cuda")
# inp = get_test_input()
print(inp.shape)
pred = model(inp.cuda(),target.cuda())
print (pred)
"""
model=model.to("cuda")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
if checkpoint!=None:
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])
    start_epoch=torch.load(checkpoint)['epoch']

# exit(0)
# model.set_cuda()

# optimizer=optimizer.cuda()
# print(len(dataloader))
# exit(0)
for param in model.parameters():
    param.requires_grad = True
count=0
for epoch in range(start_epoch,end_epoch):
    for batch_id,(imgs,target) in enumerate(dataloader):
        imgs=imgs.cuda()
        target=target.cuda()
        optimizer.zero_grad()
        loss=model(imgs,target)
        loss.backward()
        optimizer.step()
        # exit(0)
        # if count==3:
        #     exit(0)
        # count+=1
        if batch_id%10==0:
            print(epoch,batch_id,len(dataloader),loss.item(),model.losses)
            # print(model.losses)
    torch.save({
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict()},
    'checkpoint.epoch.{}.pth.tar'.format(epoch))