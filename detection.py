import torch
import numpy as np

class DetectionLayer(torch.nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer, self).__init__()
        self.anchors=anchors
        def forward(self,prediction,img_size):
            