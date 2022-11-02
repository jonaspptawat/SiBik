import torch
import torch.nn as nn

class ShuffleNetV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride has to be in between 1 and 2"

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = self.ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup // 2

        branch_main = [
                # Basic structure => Conv(1x1) > DWConv(kxk) > Conv(1x1)

                # Pointwise
                nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),

                # Depthwise
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),

                # Pointwise
                nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True)
                ]

        self.branch2 = nn.Sequential(*branch_main)

        if stride == 2:
            branch1 = [
                    # Depthwise
                    nn.Conv2d(inp, inp, ksize, stride, padding=pad, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),

                    # Pointwise
                    nn.Conv2d(inp, outputs, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(outputs),
                    nn.ReLU(inplace=True)
                    ]
            self.branch1 = nn.Sequential(*branch1)
        else:
            self.branch1 = nn.Sequential()

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)

        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)

        out = self.channel_shuffle(out, 2)
        return out

    # https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
    def channel_shuffle(self, x, groups=2):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # Reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        
        return x

class ShuffleNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]

        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU(inplace=True)
                )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(stage_names)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(
                            ShuffleNetV2Block(input_channel, output_channel, mid_channels=output_channel//2,
                                              ksize=3, stride=2)
                            )
                else:
                    stageSeq.append(
                            ShuffleNetV2Block(input_channel // 2, output_channel, mid_channels=output_channel//2,
                                              ksize=3, stride=2)
                            )
                
                input_channel = output_channel

            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

        # Load Backbone weight (Need to add "cfg" variable in __init__)
        # self.PATH = cfg.MODEL.BACKBONE
        # if len(self.PATH) > 0:
            # self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)
        
        return C1, C2, C3

 
    # def _initialize_weights(self):
    #     # Since nn.Module model are contained in the model's parameters (accessed with model.parameters() ). A state_dict is simply a Python dictionary ...
    #     # So we can directly use self(nn.Module).load_state_dict
    #     self.load_state_dict(torch.load(self.PATH), strict=True)
    #     print("Backbone Weights are successfully loaded")
