import torch
import numpy as np 
from torch import nn
import torch.nn.functional as F







class ConvEncoder(nn.Module):

    def __init__(self, out_dim: int  = 200, in_channels: int = 3) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride= 2, padding= 1), #32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride= 2, padding= 1), #16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride= 2, padding= 1), #8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride= 2, padding= 1), #4 x 4
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256*4*4, out_dim),
            nn.ReLU()
        )

    
    def forward(self, img):

        x = self.net(img)
        x = x.flatten(dim = 1)
        return self.fc(x)
    

class ConvDecoder(nn.Module):

    def __init__(self, in_dim: int = 200, out_channels: int = 3):
        super().__init__()

        self.upnet = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1), #8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride = 2, padding = 1), #16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1), #32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride = 2, padding = 1), #64 x 64
        )

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256 * 4 * 4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 256, 4, 4)
        x = self.upnet(x)
        return x 
    



class RSSM(nn.Module):

    def __init__(self,
                 action_size: int,
                 stochastic_size: int = 30,
                 deter_size: int = 200,
                 obs_feat_size: int = 200,
                 hidden: int = 200):
        super().__init__()
        self.stoch = stochastic_size
        self.deter = deter_size
        self.obs_feat_size = obs_feat_size
        self.action_size = action_size

        gru_input_size = self.stoch + action_size
        self.gru  = nn.GRUCell(gru_input_size, deter_size)

        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * stochastic_size),
        )

        self.post_net = nn.Sequential(
            nn.Linear(obs_feat_size + action_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * stochastic_size)
        )

        self.to_next_s = nn.Linear(stochastic_size, stochastic_size)

        self.register_buffer("min_std", torch.tensor(0.01))
    



    def init_state(self, batch_size: int, device: str = None):
            h = torch.zeros((batch_size, self.deter), device = device)
            s = torch.zeros((batch_size, self.deter), device = device)
            return {'h': h, 's': s}
    
    
    def prior(self, h):
        params = self.prior_net(h)

        mu_p, pre_std_p = params.split(self.stoch, dim = -1)
        std_p = F.softplus(pre_std_p) + 1e-5
        std_p = std_p.clamp(min=1e-4)

        return mu_p, std_p
