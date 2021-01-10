import torch
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self,config):
        super(MyModel, self).__init__()
        #input=1 canal pq les imatges son amb escala de grisos

        if (config["activation"]=="relu"):
            self.act = nn.ReLU()        
        else:
            self.act = nn.Tanh()

        self.conv = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1),
                                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                                nn.MaxPool2d(2),
                                self.act,
                                nn.Conv2d(64, 128, kernel_size=2, stride=1, dilation=2),
                                nn.Conv2d(128, 256, kernel_size=2, stride=2, dilation=2),
                                nn.MaxPool2d(2),
                                self.act,
                                nn.Conv2d(256, 512, kernel_size=2, stride=1, dilation=2),
                                nn.MaxPool2d(2),
                                self.act
                                )
        self.linear1 = nn.Linear(2048 , 100)
        self.linear2 = nn.Linear(100, 15)


    def forward(self, x):
        #x = x.unsqueeze(1) # single channel image
        x = self.conv(x)
        #x = x.reshape(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.act(x)
        return x
