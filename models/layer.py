import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19FeatLayer(nn.Module):
    def __init__(self, device=1):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(device)

    def forward(self, x):
        out = {}
        x = x - self.mean
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x

        return out
