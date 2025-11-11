import argparse
import os
import gymnasium as gym
import torch
from torch import nn 
import numpy as np
from memory import ReplayBuffer
from model import RSSM, RewardModel, ConvEncoder, ConvDecoder
from tqdm import tqdm
from utils import *
from planner import planner
from queue import deque
from torch.utils.tensorboard import SummaryWriter
from main import eval, populate_random


def get_parser():
    parser = argparse.ArgumentParser(description="Train RSSM on Pendulum-v1")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_train_step", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")


    parser.add_argument("--stochastic_dim", type=int, default=30, help="Stochastic latent dimension")
    parser.add_argument("--deter_dim", type=int, default=200, help="Deterministic hidden dimension")
    parser.add_argument("--hidden_dim", type=int, default=400, help="Hidden dimension for reward model")
    parser.add_argument("--obs_feat_dim", type=int, default=1024, help="Observation feature dimension")

    parser.add_argument("--C", type=int, default=100, help="training number of iteration")
    parser.add_argument("--R", type=int, default=1, help="Action repeat")
    parser.add_argument("--S", type=int, default=5, help="Seed episodes")
    parser.add_argument("--T", type=int, default=15, help="Sequence length for training")
    parser.add_argument("--L", type=int, default=50, help="Sequence length for training")
    parser.add_argument("--eps", type=float, default=0.3, help="Small gaussian exploration noise")

    parser.add_argument("--load_path", type=str, default="weights/checkpoint.pth", help="Path to load the model weights from")

    return parser


def plan(rssm_model: nn.Module,
          reward_model: nn.Module,
          R: int,
          expl_noise: float,
          buffer: ReplayBuffer,
          encoder: nn.Module,
          env : TorchImageEnv,
          device
           ):

    
    episode_states  = []
    episode_actions = []
    episode_rewards = []

    # reward_model.eval()
    # encoder.eval()
    # decoder.eval()
    # rssm_model.eval()

    obs = env.reset()
    action = None
    terminated = False
    # obs_feat_past = []
    # actions_past = []

    #episode_states.append(obs.cpu())
    

    while terminated == False:
        obs = obs.to(device)
        obs = obs.unsqueeze(0)
        obs_feat = encoder(obs)
        if action is None:
            current_state = rssm_model.init_state(obs_feat)
        else:
            current_state = rssm_model.observe_step(obs_feat, current_state['h'], current_state['s'], action.unsqueeze(0))
        # with torch.no_grad():
        #     obs_feat = encoder(obs) # shape: (1, obs_feat_dim)
        #     obs_feat_unsqueezed = obs_feat.unsqueeze(0)
        #     obs_feat_past.append(obs_feat_unsqueezed)
        #     obs_feat_past_tensor = torch.cat(obs_feat_past, dim=1)
        
        # if action is not None:
        #     actions_past.append(action.unsqueeze(0).unsqueeze(0))
        #     actions_past_tensor = torch.cat(actions_past, dim=1)
        #     with torch.no_grad():
        #         out = rssm_model.forward_observe(
        #             obs_feat_past_tensor,
        #             actions_past_tensor,
        #         )
        #         hs = out['hs']
        #         ss = out['ss']
        #         cur_state_belief = {'h':torch.unbind(hs, dim=1)[-1], 's':torch.unbind(ss, dim=1)[-1]}
        # else:
        #     with torch.no_grad():
        #         cur_state_belief = rssm_model.init_state(obs_feat)

        action = planner(rssm_model, reward_model, current_state, device = device)
        expl_noise_tensor = expl_noise * torch.randn_like(action)
        action  = action + expl_noise_tensor
        action = torch.clamp(action, min = -2.0, max= 2.0)
    
        reward = 0
        for _ in range(R):
            y, r, d, t,_ =  env.step(action.cpu().numpy())
            reward += r
            terminated = d | t 
            if terminated:
                break
        
        episode_states.append(obs[0].cpu())
        episode_actions.append(action.cpu())
        episode_rewards.append(torch.tensor(reward, dtype= torch.float32))
        obs = y

        
    episode_states.append(obs)
    

    observations_t = torch.stack(episode_states, dim = 0)
    actions_t = torch.stack(episode_actions, dim = 0)
    rewards_t  =torch.stack(episode_rewards, dim = 0)



    episode = {
        "observations": observations_t,
        "actions": actions_t,
        "rewards": rewards_t
    }
    buffer.add_episode(episode)

    return np.sum(episode_rewards)



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
    load_model(cfg.load_path, rssm_model, reward_model, encoder, decoder, device)


    populate_random(env, buffer, num_episodes = cfg.S)

    episode_rewards = deque(maxlen = 100)
    # writer = SummaryWriter(log_dir="runs/planet_pendulum")
    # save_path = "weights/"
    # os.makedirs(save_path, exist_ok= True)

    for step in tqdm(range(cfg.num_train_step), desc= f'Step:'):
        # ep_reward = plan(
        #     rssm_model,
        #     reward_model,
        #     cfg.R,
        #     cfg.eps,
        #     buffer,
        #     encoder,
        #     env,
        #     device
        # )
        # episode_rewards.append(ep_reward)
        
        # writer.add_scalar(f"Reward/mean_reward",np.mean(episode_rewards), step)
        if step % 5 == 0:
            # print(f"Mean reward: {np.mean(episode_rewards)}")
            # save_model(save_path, rssm_model, reward_model, encoder, decoder)
            reward_model.eval()
            encoder.eval()
            decoder.eval()
            rssm_model.eval()
            with torch.no_grad():
                episode, visu = eval(env, rssm_model, encoder, decoder, reward_model, device)
                save_videos(visu, save_dir= 'videos')
            buffer.add_episode(episode)
            # visualize_episode(env, rssm_model=rssm_model, reward_model = reward_model, encoder= encoder, device= device)
    return 


if __name__ == "__main__":
    parser = get_parser()
    cfg = parser.parse_args()
    main(cfg)