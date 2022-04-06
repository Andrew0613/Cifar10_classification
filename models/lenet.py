import torch
import torch.nn as nn
from models.networks import *
class LeNet(BasicNet):
    def __init__(self,opt):
        class_num = opt.output_nc
        # 定义模型的网络结构
        super().__init__(opt)
        self.conv1 = nn.Sequential(     
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2 ,stride = 2,padding=0)   
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding=0)   
        )

        self.fc1 = nn.Sequential(
            nn.Linear(576,120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84,class_num)
    def forward(self, x):
        # 定义模型前向传播的内容
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x