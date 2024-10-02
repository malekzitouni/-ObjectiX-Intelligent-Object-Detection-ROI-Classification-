import scipy.misc as misc
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
        def __init__(self,NumClasses, UseGPU=True): 
            super(Net, self).__init__()
            self.UseGPU = UseGPU # Use GPu with cuda
            self.Net = models.resnet50(pretrained=True)
            self.Net.fc=nn.Linear(2048, NumClasses)

            self.Attentionlayer = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True) 
            self.Attentionlayer.bias.data = torch.ones(self.Attentionlayer.bias.data.shape)
            self.Attentionlayer.weight.data = torch.zeros(self.Attentionlayer.weight.data.shape)

        def forward(self,Images,ROI):
                      
                InpImages = torch.autograd.Variable(torch.from_numpy(Images), requires_grad=False).transpose(2,3).transpose(1, 2).type(torch.FloatTensor)
                ROImap = torch.autograd.Variable(torch.from_numpy(ROI.astype(float)), requires_grad=False).unsqueeze(dim=1).type(torch.FloatTensor)
                if self.UseGPU == True: # Convert to GPU
                    InpImages = InpImages.cuda()
                    ROImap = ROImap.cuda()
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # Normalize image by std and mean
                x=InpImages

                x = self.Net.conv1(x) 
                AttentionMap = self.Attentionlayer(F.interpolate(ROImap, size=x.shape[2:4], mode='bilinear'))
                x = x + AttentionMap

                x = self.Net.bn1(x)
                x = self.Net.relu(x)
                x = self.Net.maxpool(x)
                x = self.Net.layer1(x)
                x = self.Net.layer2(x)
                x = self.Net.layer3(x)
                x = self.Net.layer4(x)
                x = torch.mean(torch.mean(x, dim=2), dim=2)
                x = self.Net.fc(x)
                ProbVec = F.softmax(x,dim=1) # Probability vector for all classes
                Prob,Pred=ProbVec.max(dim=1) # Top predicted class and probability
                return ProbVec,Pred