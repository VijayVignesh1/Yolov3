from dataloader import DataLoader
import torch
from model import *
from yolo import DarkNet
from matplotlib import pyplot as plt

def main():
    """ Train the Yolov3 Model """

    # If there checkpoint is already, assign checkpoint=checkpoint_file
    checkpoint=None

    # Set epochs, load the data and the trainable model
    start_epoch=0
    end_epoch=7000
    learning_rate=1e-3
    batch_size=6

    model = DarkNet()
    data=DataLoader(416,"data/train")
    dataloader=torch.utils.data.DataLoader(dataset=data,batch_size=batch_size,num_workers=0,shuffle=True)
    model=model.to("cuda")
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

    # If there's a checkpoint, load its values
    if checkpoint!=None:
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])
        start_epoch=torch.load(checkpoint)['epoch']

    for param in model.parameters():
        param.requires_grad = True
    count=0
    x_y=[]
    w_h=[]
    conf_loss=[]
    final_loss=[]

    # Train the model
    print("Starting Training..")

    for epoch in range(start_epoch,end_epoch):
        print("------------------------------------------------------------------------------------------------------------")
        for batch_id,(imgs,target) in enumerate(dataloader):
            imgs=imgs.cuda()
            target=target.cuda()
            optimizer.zero_grad()
            loss=model(imgs,target)
            loss.backward()
            optimizer.step()
            if batch_id%10==0:
                print("Epoch %d/%d || Batch %d || Overall Loss %.2f || X-Loss %.2f || Y-Loss %.2f || W-Loss %.2f || H-Loss %.2f" %(epoch, 
                                    end_epoch, batch_id, loss.item(), model.losses[0], model.losses[1], model.losses[2], model.losses[3]))
        x_y.append(model.losses[0]+model.losses[1])
        w_h.append(model.losses[2]+model.losses[3])
        conf_loss.append(model.losses[4])
        final_loss.append(loss.item())

    # Plot the graph to check if the loss is decreasing through the epochs
    
    # X-Y Loss
    plt.plot(x_y,label='X and Y')
    plt.savefig('x-y-loss.png')
    plt.close()

    # W-H Loss
    plt.plot(w_h,label='W and H')
    plt.savefig('w-h-loss.png')
    plt.close()

    # Confidence Loss
    plt.plot(conf_loss,label='Conf')
    plt.savefig('conf-loss.png')
    plt.close()

    # Overall Loss
    plt.plot(final_loss,label='Loss')
    plt.savefig('final-loss.png')
    plt.show()
    plt.close()

    # Save the model as checkpoint
    torch.save({
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict()},
    'checkpoints/checkpoint.epoch.{}.pth.tar'.format(epoch))

if __name__=="__main__":
    main()
