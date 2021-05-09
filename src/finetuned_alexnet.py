import torch
import torch.nn as nn
from torchvision.models import alexnet

class FinetunedAlexNet(nn.Module):
  def __init__(self):
    super(FinetunedAlexNet, self).__init__()
    # super().__init__()

    self.model = alexnet(pretrained=True)
    self.model.classifier[6] = nn.Linear(4096, 200)
    
    for param in self.model.features.parameters():
        param.requires_grad = False
    for param in self.model.avgpool.parameters():
        param.requires_grad = False

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   output: the output (raw scores) of the net [Dim: (N,200)]
    '''

    output = self.model(x)
    return output

  def unfreezeAll(self):
    for param in self.model.parameters():
        param.requires_grad = True
