import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fa import FALayer

class CIFAR101(nn.Module):
    """ Arch 1 from Moskovitz et al.
    Input: 24x24x3
    """
    def __init__(self, num_classes=10, learn="BP", fa_scale=0.1):
        super().__init__()
        assert learn in ['BP', 'FA', 'FA_toeplitz', 'FA_uSF_init', 'FA_uSF_sn']
        self.learn = learn
        # 64,20,20
        self.conv1 = nn.Conv2d(3, 64, 5)
        # after pool is 64,10,10. Then 64, 6, 6
        self.conv2 = nn.Conv2d(64, 64, 5)
        # after pool becomes 64,3,3
        self.fc1 = nn.Linear(576, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

        if learn != "BP":
            mode_map = {
                'FA': (False, 'fa'),
                'FA_toeplitz': (True, 'fa'),
                'FA_uSF_init': (True, 'usf_init'),
                'FA_uSF_sn': (True, 'usf_sn'),
            }
            use_toeplitz, fa_mode = mode_map[learn]

            self.conv2 = FALayer(self.conv2, fa_scale, use_toeplitz, mode=fa_mode)
            self.fc1 = FALayer(self.fc1, fa_scale, use_toeplitz, mode=fa_mode)
            self.fc2 = FALayer(self.fc2, fa_scale, use_toeplitz, mode=fa_mode)
            self.fc3 = FALayer(self.fc3, fa_scale, use_toeplitz, mode=fa_mode)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



