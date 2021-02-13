import torch
from torch import nn
import torchvision
import torch.nn.functional as f
#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
### Try out some transformer based arch 
class Resnet101Encoder(nn.Module):
    """
    Resnet101 based encoder Encoder.
    """
    def __init__(self, encoded_image_size=14):
        super(Resnet101Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        modules = list(resnet.children())[:-2] # Remove linear and pool layers (since we're not doing classification)
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)) # Resize image to fixed size to allow input images of variable size
        self.fine_tune()
    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)# (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = f.normalize(out, p=2, dim=2)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
