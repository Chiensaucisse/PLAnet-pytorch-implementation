import gymnasium as gym
import cv2
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

def atanh(x: torch.Tensor):
    return 0.5 * torch.log((1 + x) / (1 - x))

def reparameterize(mu: torch.Tensor, std: torch.Tensor):
    eps = torch.randn_like(std)

    return mu + eps * std


def kl_divergence_diag(mu_q: torch.Tensor, std_q: torch.Tensor, mu_p: torch.Tensor, std_p: torch.Tensor ):

    var_q = std_q.pow(2)
    var_p = std_p.pow(2)
    k = mu_q.shape[-1]
    kl = (1/2) * (
        ((var_q + (mu_q - mu_p).pow(2)) / var_p).sum(-1)
        - k
        + 2 * (torch.log(std_p).sum(-1) - torch.log(std_q).sum(-1))
    )

    return kl 

def to_tensor_obs(image: np.ndarray, size: tuple):
    

    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    return image

def preprocess_img(image: torch.Tensor, depth: int = 8):
    image.div_(2 ** (8 - depth)).floor_().div_(2 ** depth).sub_(0.5)
    image.add_(torch.randn_like(image).div_(2 ** depth)).clamp_(-0.5, 0.5)

def postprocess_img(image, depth):

    image = np.floor((image + 0.5) * 2 ** depth)
    return np.clip(image * 2**(8 - depth), 0, 2**8 - 1).astype(np.uint8)

def plot_reconstructed(renconstructed_img, tgt_image):

    idx = 0
    recon_seq = renconstructed_img[idx].detach().cpu()
    target_seq = tgt_image[idx].detach().cpu()
    L = recon_seq.shape[0]

   
    fig, axs = plt.subplots(2, L, figsize=(3 * L, 6))

    for t in range(L):
     
        recon_np = recon_seq[t].permute(1, 2, 0).numpy()
        target_np = target_seq[t].permute(1, 2, 0).numpy()
        
    
        axs[0, t].imshow(target_np)
        axs[0, t].set_title(f"Target t={t}")
        axs[0, t].axis("off")
        
        axs[1, t].imshow(recon_np)
        axs[1, t].set_title(f"Recon t={t}")
        axs[1, t].axis("off")

    plt.tight_layout()
    plt.savefig(f"debug/reconstruction_seq_batch{idx}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)



def compute_losses(rssm_out: dict,
                   observation_images: torch.Tensor,
                   rewards_gt: torch.Tensor,
                   decoder,
                   reward_model,
                   kl_free_nats = 3.0,
                   kl_scale = 1.0,
                   recon_weight = 1.0,
                   reward_weight = 1.0):
    
    
    hs = rssm_out['hs'] # (B, L, H)
    ss = rssm_out['ss'] # (B, L, S)
    
    latent_feats = torch.cat([hs, ss], dim = -1) # (B, L, H + S)
    B, L, D = latent_feats.shape
    squeeze_latent = latent_feats.view(B*L, -1)
    squeeze_decoded_obs = decoder(squeeze_latent)
    decoded_obs = squeeze_decoded_obs.view(B, L, 3, 64, 64)
 
    reconstructed = decoded_obs # torch.sigmoid(decoded_obs)

    # if True:
    #     plot_reconstructed(reconstructed, observation_images)

    reconstruction_loss  = F.mse_loss(reconstructed, observation_images[:,1:], reduction = 'none').sum([2,3,4]).mean()
    reconstruction_loss *=  recon_weight

    reward_preds = reward_model(squeeze_latent).view(B,L)


    reward_loss = F.mse_loss(reward_preds, rewards_gt, reduce= 'mean') * reward_weight

    mu_qs = rssm_out['mu_qs']
    std_qs = rssm_out['std_qs']
    mu_ps = rssm_out['mu_ps']
    std_ps = rssm_out['std_ps']

    kl  = kl_divergence_diag(mu_qs, std_qs, mu_ps, std_ps)
    kl = torch.clamp(kl - kl_free_nats, min = 0.0)
    kl_loss = kl.mean() * kl_scale

    total_loss = reconstruction_loss + reward_loss + kl_loss

    return {
        "total_loss": total_loss,
        "recon_loss": reconstruction_loss.item(),
        "reward_loss": reward_loss.item(),
        "kl_loss": kl_loss.item()
    }



class TorchImageEnv:

    def __init__(self, env, img_size: tuple = (64,64)):

        self.env = gym.make(env, render_mode = "rgb_array")
        self.img_size = img_size


    def reset(self):

        self.env.reset()
        x = to_tensor_obs(self.env.render(), size = self.img_size)

        return x
    
    def step(self, action: np.ndarray):

        _, r, d, i, f  = self.env.step(action)
        x = to_tensor_obs(self.env.render(), size = self.img_size)
        # preprocess_img(x)

        return x , r , d , i, f
    
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_size(self):
        return self.img_size

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    def sample_random_action(self):
        return torch.tensor(self.env.action_space.sample())
    


def save_model(save_root: str,
               rssm_model: nn.Module,
               reward_model: nn.Module,
               encoder: nn.Module,
               decoder: nn.Module) -> None:
    
    checkpoint = {
        'rssm_model_state_dict': rssm_model.state_dict(),
        'reward_model_state_dict': reward_model.state_dict(),
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict()
    }
    save_path = Path(save_root) / 'checkpoint.pth'
    torch.save(checkpoint, save_path)



if __name__ == "__main__":

    env = TorchImageEnv('Pendulum-v1')

    x  = env.reset()

    print(x.shape)