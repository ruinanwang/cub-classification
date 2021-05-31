import torch
import torch.nn as nn
from torchvision.models import alexnet
from torchvision.models import resnet18
from torchvision.models import inception_v3

class FinetunedAlexNet(nn.Module):
    def __init__(self):
        super(FinetunedAlexNet, self).__init__()
        # super().__init__()

        self.model = alexnet(pretrained=True)
        classifier = list(self.model.classifier.children())[:-1]
        classifier.append(nn.Linear(4096, 312))
        classifier.append(nn.Tanh())
        self.model.classifier = nn.Sequential(*classifier)
#         self.model.classifier[6] = nn.Linear(4096, 312)
#         self.model.classifier.append(nn.Tanh())

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
            
class FinetunedAlexNet1(nn.Module):
    def __init__(self, num_attributes):
        super(FinetunedAlexNet1, self).__init__()

        self.model = alexnet(pretrained=True)
        classifier = list(self.model.classifier.children())[:-1]
        classifier.append(nn.Linear(4096, num_attributes))
        self.model.classifier = nn.Sequential(*classifier)
        self.fc_list = []
        for i in range(num_attributes):
            self.fc_list.append(nn.Linear(num_attributes, 1))

#         for param in self.model.features.parameters():
#             param.requires_grad = False
#         for param in self.model.avgpool.parameters():
#             param.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   output: the output (raw scores) of the net [Dim: (N,200)]
        '''

        output = self.model(x)
        output_list = []
        for fc in self.fc_list:
            output_list.append(nn.Sigmoid()(fc.cuda()(output)))
        return output_list

    def unfreezeAll(self):
        for param in self.model.parameters():
            param.requires_grad = True
        
class FinetunedResNet(nn.Module):
  
    # constructor
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        last_layer_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_inputs, 200)  
#         self.model.to(device)

    def forward(self, x):
        x = self.model.forward(x)
        return x

class FinetunedResNet1(nn.Module):
  
    # constructor
    def __init__(self, num_attributes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        last_layer_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_inputs, num_attributes)
        self.fc_list = []
        for i in range(num_attributes):
            self.fc_list.append(nn.Linear(num_attributes, 1))
#         self.model.to(device)

    def forward(self, x):
        output = self.model.forward(x)
        output_list = []
        for fc in self.fc_list:
            output_list.append(nn.Sigmoid()(fc.cuda()(output)))
#         print(output_list)
        return output_list

class FinetunedResNet2(nn.Module):
  
    # constructor
    def __init__(self, num_attributes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        last_layer_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_inputs, num_attributes)  

    def forward(self, x):
        x = self.model.forward(x)
        x = nn.Sigmoid()(x)
        return x
    
class FinetunedInceptionV3(nn.Module):
    def __init__(self):
        super(FinetunedInceptionV3, self).__init__()
        self.model = inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, 200)


#         for param in self.model.features.parameters():
#             param.requires_grad = False
#         for param in self.model.avgpool.parameters():
#             param.requires_grad = False

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

            
class FinetunedInceptionV3_1(nn.Module):
    def __init__(self, num_attributes):
        super(FinetunedInceptionV3_1, self).__init__()
        self.model = inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, num_attributes)
        self.fc_list = []
        for i in range(num_attributes):
            self.fc_list.append(nn.Linear(num_attributes, 1))


#         for param in self.model.features.parameters():
#             param.requires_grad = False
#         for param in self.model.avgpool.parameters():
#             param.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   output: the output (raw scores) of the net [Dim: (N,200)]
        '''

        output = self.model(x)
        output_list = []
        if self.training:
            for fc in self.fc_list:
                output_list.append(nn.Sigmoid()(fc.cuda()(output.logits)))
        else:
            for fc in self.fc_list:
                output_list.append(nn.Sigmoid()(fc.cuda()(output)))
        return output_list

    def unfreezeAll(self):
        for param in self.model.parameters():
            param.requires_grad = True
            
class FinetunedInceptionV3_2(nn.Module):
    def __init__(self, num_attributes):
        super(FinetunedInceptionV3_2, self).__init__()
        self.model = inception_v3(pretrained=True)
        last_layer_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_inputs, num_attributes)  

    def forward(self, x):
        output = self.model(x)
        if self.training:
            output = nn.Sigmoid()(output.logits)
        else:
            output = nn.Sigmoid()(output)
        return output
