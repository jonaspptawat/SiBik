# This code contains code from these github
# https://github.com/dog-qiuqiu/FastestDet/blob/main/module/custom_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.conv1x1 = nn.Sequential(
                nn.Conv2d(inp, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True)
                )
    
    def forward(self, x):
        return self.conv1x1(x)

class Head(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        
        self.conv = nn.Sequential(
                # Return the same spatial size and feature size (Depthwise conv)
                nn.Conv2d(inp, inp, 5, 1, 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # Pointwise
                nn.Conv2d(inp, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out)
                )

    def forward(self, x):
        return self.conv(x)

class DetHead(nn.Module):
    def __init__(self, inp, n_classes):
        super().__init__()

        self.conv = Conv1x1(inp, inp)

        # This for anchor free
        self.obj = Head(inp, 1) 
        self.reg = Head(inp, 4)
        self.cls = Head(inp, n_classes)
        
        # For obj
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)

        obj = self.sigmoid(self.obj(x))
        reg = self.reg(x)
        cls = self.cls(x)
        
        # This return [obj, x, y, w, h, cls...]
        return torch.cat((obj, reg, cls), dim=1)

class FPN(nn.Module):
    def __init__(self, c1_depth, c2_depth, c3_depth, out_depth):
        super().__init__()

        self.conv_c1 = nn.Sequential(
                nn.Conv2d(c1_depth, out_depth, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_depth),
                nn.ReLU(inplace=True)
                )
        
        self.conv_c2 = nn.Sequential(
                nn.Conv2d(c2_depth, out_depth, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_depth),
                nn.ReLU(inplace=True)
                )

        self.conv_c3 = nn.Sequential(
                nn.Conv2d(c3_depth, out_depth, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_depth),
                nn.ReLU(inplace=True)
                )

    def forward(self, C1, C2, C3):
        # Get S1 by concatenating interpolated C2 of factor of 2 and C1 together
        c2_inter = F.interpolate(C2, scale_factor=2)
        c1_c2_cat = torch.cat([c2_inter, C1], dim=1)
        S1 = self.conv_c1(c1_c2_cat)

        # Get S2 by concatenating interpolated C3 of factor of 2 and C2 together
        c3_inter = F.interpolate(C3, scale_factor=2)
        c2_c3_cat = torch.cat([c3_inter, C2], dim=1)
        S2 = self.conv_c2(c2_c3_cat)

        S3 = self.conv_c3(C3)

        return S1, S2, S3

class SPP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.convpw = Conv1x1(inp, out)

        self.S1 = nn.Sequential(
                nn.Conv2d(out, out, 5, 1, 2, groups=out, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True)
                )

        self.S2 = nn.Sequential(
                nn.Conv2d(out, out, 5, 1, 2, groups=out, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),

                nn.Conv2d(out, out, 5, 1, 2, groups=out, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True)
                )

        self.S3 = nn.Sequential(
                nn.Conv2d(out, out, 5, 1, 2, groups=out, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),

                nn.Conv2d(out, out, 5, 1, 2, groups=out, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),

                nn.Conv2d(out, out, 5, 1, 2, groups=out, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True)
                )
        
        self.output = nn.Sequential(
                nn.Conv2d(out * 3, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out)
                ) 

        self.rl = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.convpw(x)

        y1 = self.S1(x)
        y2 = self.S2(x)
        y3 = self.S3(x)

        y = torch.cat([y1, y2, y3], dim=1)
        y = self.rl(x + self.output(y))
        return y
