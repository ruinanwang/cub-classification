import torch
import torch.nn as nn
from torchvision.models import alexnet
from torchvision.models import resnet18
from torchvision.models import vgg16
from torchvision.models import densenet161

class FinetunedAlexNet(nn.Module):
  def __init__(self):
    super(FinetunedAlexNet, self).__init__()
    # super().__init__()

    self.model = alexnet(pretrained=True)
#     self.model.classifier[6] = nn.Linear(4096, 200)
    classifier = list(self.model.classifier.children())[:-1]
    classifier.append(nn.Linear(4096, 200))
#     classifier[0] = nn.Dropout(0.8)
#     classifier[3] = nn.Dropout(0.8)
    self.model.classifier = nn.Sequential(*classifier)
    
#     for param in self.model.features.parameters():
#         param.requires_grad = False
#     for param in self.model.avgpool.parameters():
#         param.requires_grad = False

  def forward(self, x: torch.tensor) -> torch.tensor:
    output = self.model(x)
    return output

  def unfreezeAll(self):
    for param in self.model.parameters():
        param.requires_grad = True
        
class FinetunedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        last_layer_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_inputs, 200)  

    def forward(self, x):
        x = self.model.forward(x)
        return x

class FinetunedVggNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg16(pretrained=True)
        classifier = list(self.model.classifier.children())[:-1]
        classifier.append(nn.Linear(4096, 200))
        self.model.classifier = nn.Sequential(*classifier)
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.avgpool.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model.forward(x)
        return x

class FinetunedDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = densenet161(pretrained=True)
        self.model.classifier = nn.Linear(2208, 200)
        for param in self.model.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model.forward(x)
        return x