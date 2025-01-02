from torch import nn
from efficientnet_pytorch import EfficientNet

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        # advprop: Whether to load pretrained weights trained with advprop (valid when weights_path is None).

    def forward(self,x):
        x=self.net(x)
        return x