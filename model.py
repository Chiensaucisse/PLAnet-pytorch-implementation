import torch
import numpy as np 
from torch import nn
import torch.nn.functional as F
from utils import reparameterize
from typing import Tuple, List





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
            nn.Linear(obs_feat_size + deter_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * stochastic_size)
        )

        self.to_next_s = nn.Linear(stochastic_size, stochastic_size)

        self.register_buffer("min_std", torch.tensor(0.01))
    



    def init_state(self, batch_size: int, device: str = None) -> dict:
            h = torch.zeros((batch_size, self.deter), device = device)
            s = torch.zeros((batch_size, self.deter), device = device)
            return {'h': h, 's': s}
    
    
    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor]:
        params = self.prior_net(h)

        mu_p, pre_std_p = params.split(self.stoch, dim = -1)
        std_p = F.softplus(pre_std_p) + 1e-5
        std_p = std_p.clamp(min=1e-4)

        return mu_p, std_p

    def posterior(self, obs_feat: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor]:
        params = self.post_net(obs_feat, h)
        mu_q, pre_std_q = params.split(self.stoch, dim  = -1)
        std_q = F.softplus(pre_std_q) + 1e-5
        std_q = std_q.clamp(min=1e-4)

        return mu_q, std_q
    
    def observe_step(self, 
                     obs_feat: torch.Tensor, 
                     prev_h: torch.Tensor, 
                     prev_s: torch.Tensor,
                     action: torch.Tensor) -> dict:
        
        gru_input = torch.cat([prev_s, action], dim = 1)
        h = self.gru(gru_input, prev_h)
        mu_p, std_p = self.prior(h)
        mu_q, std_q = self.posterior(obs_feat, h)

        s = reparameterize(mu_q, std_q)
        s_embed = F.relu(self.to_next_s(s))

        out = {
            'mu_p': mu_p,
            'std_p': std_p,
            'mu_q': std_q,
            'std_q': std_q,
            's': s,
            's_embed': s_embed,
            'h': h
        }

        return out
    
    def imagine_step(self,
                     prev_h: torch.Tensor,
                     prev_s: torch.Tensor,
                     action: torch.Tensor,
                     ) -> dict:
        gru_input = torch.cat([prev_s, action], dim = -1)
        h = self.gru(gru_input, prev_h)
        mu_p, std_p = self.prior(h)
        s = reparameterize(mu_p, std_p)
        s_embed = self.to_next_s(s)
        out =  {
            'mu_p': mu_p,
            'std_p': std_p,
            's': s,
            's_embed': s_embed,
            'h': h
        }
        return out

    def forward_observe(self, 
                        obs_feats: torch.Tensor,
                        actions: torch.Tensor,
                        init_state: torch.Tensor = None) -> dict:
                
                B, L, _ = actions.shape
                if init_state is None:
                     init_state = self.init_state(B, device = actions.device)
                
                prev_h = init_state['h']
                prev_s = init_state['s']

                mu_ps = []
                std_ps = []
                mu_qs = []
                std_qs = []
                ss = []
                s_embeds = []
                hs = []

                for l in range(L):
                     obs_feat = obs_feats[:, l]
                     action = actions[:, l]
                     out = self.observe_step(obs_feat, prev_h, prev_s, action)
                     mu_ps.append(out['mu_p'])
                     std_ps.append(out['std_p'])
                     mu_qs.append(out['mu_q'])
                     std_qs.append(out ['std_q'])
                     ss.append(out['s'])
                     
                     h = out['h']
                     s = out['s']
                     hs.append(h)
                     s_embeds.append(s)

                     prev_h = h
                     prev_s = s
                    
                out =  {
                     'mu_ps': torch.stack(mu_ps, dim = 1),
                     'std_ps': torch.stack(std_ps, dim = 1),
                     'mu_qs': torch.stack(mu_qs, dim = 1),
                     'std_qs': torch.stack(std_qs, dim = -1),
                     'ss': torch.stack(ss, dim = 1),
                     'hs': torch.stack(hs, dim = 1)
                }

                return out
    
    def imagine_ahead(self,
                      actions: torch.Tensor,
                      init_state: torch.Tensor = None) -> dict:
        
        B, L, _ = actions.shape
        if init_state is None:
                init_state = self.init_state(B, device = actions.device)
        
        prev_h = init_state['h']
        prev_s = init_state['s']

        mu_ps = []
        std_ps = []
        ss = []
        s_embeds = []
        hs = []

        for l in range(L):

                action = actions[:, l]
                out = self.imagine_step(prev_h, prev_s, action)
                mu_ps.append(out['mu_p'])
                std_ps.append(out['std_p'])
                ss.append(out['s'])
                
                h = out['h']
                s = out['s']
                hs.append(h)
                s_embeds.append(s)

                prev_h = h
                prev_s = s
            
        out =  {
                'mu_ps': torch.stack(mu_ps, dim = 1),
                'std_ps': torch.stack(std_ps, dim = 1),
                'ss': torch.stack(ss, dim = 1),
                'hs': torch.stack(hs, dim = 1)
        }

        return out
    



class RewardModel(nn.Module):
     
    def __init__(self, in_dim: int, hidden_dim: int = 200) -> None:
          self.mlp = nn.Sequential(
               nn.Linear(in_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, 1)
          )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.mlp(x)
        return x 