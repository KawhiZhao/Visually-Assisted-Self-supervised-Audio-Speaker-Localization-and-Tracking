import torch
from torch import nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
import numpy as np
import pdb


class audioNetwork_multi_task(nn.Module):
    def __init__(self):
        super().__init__()
        self.cam1_pos = np.array([-1.57, 2.01, 1.43])
        self.cam2_pos = np.array([-1.57, -2.32, 1.19])
        self.cam3_pos = np.array([-0.315, -3.105, 1.35])
        self.dropout_rate = 0.1
        # encoder
        self.convs = nn.Sequential(
            nn.Conv2d(1, 2, [3, 3], padding=1),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 4, [3, 3], padding=1),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, [3, 3], padding=1),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, [3, 3], padding=1),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, [3, 3], padding=1),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, [3, 3], padding=1),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, [3, 3], padding=1),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
            nn.BatchNorm2d(128),
        )


        # decoder for localization
        self.fc_group1 = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.Linear(256, 128),
            nn.Linear(131, 64),
            nn.Linear(64, 32),
        )
        self.fc_group2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Linear(16, 2),
        )
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        # decoder for semantic segmentation
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(131, 64, (16, 16), (2, 2)),
            nn.ConvTranspose2d(64, 32, (16, 16), (5, 5), padding=(1,1)),
            nn.ConvTranspose2d(32, 2, (8, 24), (4, 3)),
        )

    def forward(self, x, seq_names, device):
        x = self.convs(x)
        
        camera_meta = np.zeros((x.shape[0], 3))
        camera_meta = torch.tensor(camera_meta)
        
        for i, seq_name in enumerate(seq_names):
            if 'cam1' in seq_name:
                camera_meta[i, :] = torch.tensor(self.cam1_pos)
            elif 'cam2' in seq_name:
                camera_meta[i, :] = torch.tensor(self.cam2_pos)
            elif 'cam3' in seq_name:
                camera_meta[i, :] = torch.tensor(self.cam3_pos)
        camera_meta = self.sig1(camera_meta)
        camera_meta = camera_meta.to(device)
        camera_meta = camera_meta.type(torch.cuda.FloatTensor)
        camera_meta = camera_meta.unsqueeze(1)
        camera_meta = camera_meta.unsqueeze(1)
        x3 = x.permute(0, 2, 3, 1)
        # pdb.set_trace()
        x3 = torch.cat([x3, camera_meta], dim=3)

        x1 = x3.squeeze()
        x1 = self.fc_group1(x1)
        
        x1 = self.fc_group2(x1)
        x1 = self.sig2(x1)
        # x[:, 0] = x[:, 0] / 360
        # x[:, 1] = x[:, 1] / 288
        x4 = x3.permute(0, 3, 1, 2)
        x2 = self.deconv(x4)

        return x1, x2