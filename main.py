import argparse
import gymnasium as gym
import torch
from torch import nn 
import numpy
from memory import ReplayBuffer
from model import RSSM, RewardModel, ConvEncoder, ConvDecoder
from tqdm import tqdm
from utils import *



def get_parser():
    parser = argparse.ArgumentParser(description="Train RSSM on Pendulum-v1")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_train_step", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")


    parser.add_argument("--stochastic_dim", type=int, default=30, help="Stochastic latent dimension")
    parser.add_argument("--deter_dim", type=int, default=200, help="Deterministic hidden dimension")
    parser.add_argument("--hidden_dim", type=int, default=400, help="Hidden dimension for reward model")
    parser.add_argument("--obs_feat_dim", type=int, default=1024, help="Observation feature dimension")

    parser.add_argument("--C", type=int, default=100, help="Chunk length for RSSM training")
    parser.add_argument("--R", type=int, default=4, help="Number of rollout steps")
    parser.add_argument("--S", type=int, default=5, help="Seed episodes")
    parser.add_argument("--T", type=int, default=15, help="Sequence length for training")
    parser.add_argument("--L", type=int, default=50, help="Sequence length for training")

    return parser

def populate_random(env: TorchImageEnv, buffer: ReplayBuffer,  num_episodes: int = 10):


    

    for episode in tqdm(range(num_episodes), desc="Populating with random policy"):
        state  = env.reset()
        terminated  = False
        observations_list = []
        actions_list = []
        rewards_list = []
        while terminated == False:

            # Random action
            action = env.sample_random_action()

        
            next_state, reward, done, truncated, info = env.step(action)
            observations_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)

            state = next_state

            if done or truncated:
                terminated = True
                observations_list.append(state)
    
        observations_t = torch.stack(observations_list, dim = 0)
        actions_t = torch.stack(actions_list, dim = 0)
        rewards_t  = torch.stack(rewards_list, dim = 0)

        episode = {
            "observations": observations_t,
            "actions": actions_t,
            "rewards": rewards_t
        }
        buffer.add_episode(episode)

def fit_rssm(
        rssm_model: RSSM,
        reward_model: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        batch: dict,
        optimizer
    ) -> dict:


    observations_image = batch['observations']
    actions =  batch['actions']
    rewards = batch['rewards']
    B, L, C, H, W = observations_image.shape
    observations_image_squeezed = observations_image.view(B * L, C, H, W)
    obs_feat = encoder(observations_image_squeezed)
    obs_feat = obs_feat.view(B, L, -1)

    rssm_out = rssm_model.forward_observe(
        obs_feats= obs_feat,
        actions= actions,
        init_state = None,
    )

    losses = compute_losses(
        rssm_out,
        observations_image,
        rewards,
        decoder,
        reward_model
    )

    total_loss = losses['total_loss']
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    losses['total_loss'] = total_loss.detach()

    return losses


def train(rssm_model: nn.Module,
          reward_model: nn.Module,
          C :int, 
          R: int,
          T: int,
          L: int,
          batch_size: int,
          buffer: ReplayBuffer,
          encoder: nn.Module,
          decoder: nn.Module,
          optim,
           ):



    episode_states  = []
    episode_actions = []
    episodes_rewards = []


    batch = buffer.sample(batch_size, chunk_length= L)
    losses = fit_rssm(rssm_model,
            reward_model,
            encoder,
            decoder,
            batch, 
            optim,
            )
    print(losses)
    exit()

    x = env.reset()
    terminated = False

    while terminated == False:
        action = planner(rssm_model,        # TO DO
                        ...)
        expl_noise = torch.normal() # to do
        action  = action + expl_noise

        reward = 0
        for _ in range(R):
            y, r, d, t,_ =  env.step(action)
            reward += R
            x  = y
            terminated = d | t 
            if terminated:
                break
        next_state  = x

        episode_states.append(x)
        episode_actions.append(action)
        episodes_rewards.append(reward)

        if terminated:
            episode_states.append(next_state)
    

    observations_t = torch.stack(episode_states, dim = 0)
    actions_t = torch.stack(episode_actions, dim = 0)
    rewards_t  =torch.stack(episodes_rewards, dim = 0)

    episode = {
        "observations": observations_t,
        "actions": actions_t,
        "rewards": rewards_t
    }
    buffer.add_episode(episode)



def main(cfg):

    env = TorchImageEnv('Pendulum-v1')
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    action_size = env.action_size
    max_capacity = 500
    buffer = ReplayBuffer(capacity=max_capacity, device= device)

    rssm_model = RSSM(action_size= action_size, stochastic_size= cfg.stochastic_dim,
                      deter_size = cfg.deter_dim, obs_feat_size= cfg.obs_feat_dim,
                      hidden= cfg.hidden_dim).to(device)
    reward_model  = RewardModel(in_dim = cfg.stochastic_dim  + cfg.deter_dim, hidden_dim = cfg.hidden_dim).to(device)
    encoder = ConvEncoder(out_dim = cfg.obs_feat_dim).to(device)
    decoder = ConvDecoder(in_dim = cfg.stochastic_dim + cfg.deter_dim).to(device)

    optimizer = torch.optim.Adam(rssm_model.parameters(), lr = cfg.lr)


    populate_random(env, buffer, num_episodes = cfg.S)

    for step in range(cfg.num_train_step):
        train(
            rssm_model,
            reward_model,
            cfg.C,
            cfg.R,
            cfg.T,
            cfg.L,
            cfg.batch_size,  
            buffer,
            encoder,
            decoder,
            optimizer,
        )



    return 




if __name__ == "__main__":
    parser = get_parser()
    cfg = parser.parse_args()
    main(cfg)