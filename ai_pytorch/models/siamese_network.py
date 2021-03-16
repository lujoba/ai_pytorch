import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = models.resnet18(pretrained=True)

        self.cnn1.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 8))

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseVector(nn.Module):

    def __init__(self):
        super(SiameseVector, self).__init__()
        self.cnn1 = models.resnet18(pretrained=False)

        self.cnn1.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 8))

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1):
        output1 = self.forward_once(input1)
        return output1
