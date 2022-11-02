import torch
import torch.nn as nn
from .backbone import get_backbone

import sys
sys.path.append(".")
sys.path.append("./detector")

from layers import DetHead, FPN, SPP

class Detector(nn.Module):
    def __init__(self, classes):
        super().__init__()

        n_fpn_features = 128
        n_spp_features = 96
        stage_channels = [48, 96, 192]

        self.backbone = get_backbone()

        self.fpn = FPN(
                stage_channels[0] + stage_channels[1],
                stage_channels[1] + stage_channels[-1],
                stage_channels[-1],
                n_fpn_features
                )

        self.spp = SPP(n_fpn_features * 3, n_spp_features)
        self.det_head = DetHead(n_spp_features, classes)

        # Utils
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # Get C1, C2, and C3 from backbone
        C1, C2, C3 = self.backbone(x)

        # Pass backbone's outputs to FPN
        S1, S2, S3 = self.fpn(C1, C2, C3)
        S3 = self.upsample(S3)
        S1 = self.avg_pool(S1)
        S = torch.cat([S1, S2, S3], dim=1)

        # Pass FPN's output to SPP
        y = self.spp(S)

        return self.det_head(y) # Return detection's head

if __name__ == "__main__":

    detector = Detector(2)
    from torchsummary import summary
    import time
    sample = torch.rand((20, 3, 416, 416))
    start = time.perf_counter()
    result = detector(sample)
    end = time.perf_counter()
    time_used = end - start # In Sec
    summary(detector)
    print(f"Time used: {time_used} sec")
