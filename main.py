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

    return parser

def visualize_episode(env: TorchImageEnv, rssm_model: nn.Module, reward_model: nn.Module, encoder: nn.Module, device = None, R: int = 4) -> None:
    x = env.reset()
    env_human = gym.make('Pendulum-v1', render_mode = "human")
    _,_ = env_human.reset()
    x = x.to(device)
    x = x.unsqueeze(0)
    obs_feat = encoder(x)
    terminated = False

    while terminated == False:
        action = planner(rssm_model, reward_model, obs_feat,  device = device)
        action = torch.clamp(action, min = -2.0, max= 2.0)
    
        reward = 0
        for _ in range(R):
            env_human.render()
            y, r, d, t,_ =  env.step(action.cpu().numpy())
            env_human.step(action.cpu().numpy())
            
            reward += r
            x  = y
            terminated = d | t 
            if terminated:
                break
    env_human.close()
    


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
    obs_feat = encoder(observations_image_squeezed) # shape: (B*L, obs_feat_dim)
    obs_feat = obs_feat.view(B, L, -1)

    rssm_out = rssm_model.forward_observe(
        obs_feats= obs_feat,
        actions= actions,
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

    # losses['total_loss'] = total_loss.detach()

    return losses


def train(rssm_model: nn.Module,
          reward_model: nn.Module,
          C :int, 
          R: int,
          T: int,
          L: int,
          expl_noise: float,
          batch_size: int,
          buffer: ReplayBuffer,
          encoder: nn.Module,
          decoder: nn.Module,
          optim,
          env : TorchImageEnv,
          device
           ):
    '''
    Train RSSM model for one iteration

    Args:
        rssm_model (nn.Module): The RSSM model to be trained
        reward_model (nn.Module): The reward model
        C (int): Update steps
        R (int): Action repeat
        T (int): Time horizon for planning
        L (int): Chunk length
        batch_size (int): Batch size for training
        buffer (ReplayBuffer): Replay buffer to sample training data from
        encoder (nn.Module): Encoder model for observations
        decoder (nn.Module): Decoder model for observations
        optim: Optimizer for training the models
    Returns:
        None
    '''

    
    episode_states  = []
    episode_actions = []
    episode_rewards = []


    for _ in range(C):
        batch = buffer.sample(batch_size, chunk_length= L)
        reward_model.train()
        encoder.train()
        decoder.train()
        rssm_model.train()
        losses = fit_rssm(rssm_model,
                reward_model,
                encoder,
                decoder,
                batch, 
                optim,
                )

    obs = env.reset()
    action = None
    terminated = False

    while terminated == False:
        obs = obs.to(device)
        obs = obs.unsqueeze(0)
        obs_feat = encoder(obs)
        if action is None:
            current_state = rssm_model.init_state(obs_feat)
        else:
            current_state = rssm_model.observe_step(obs_feat, current_state['h'], current_state['s'], action.unsqueeze(0))

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

    return losses, np.sum(episode_rewards)


def eval(
        env : TorchImageEnv,
        rssm_model: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        reward_model: nn.Module,
        device = 'cpu',

):
    
    obs = env.reset()
    frames = []
    action = None
    terminated = False
    predicted_rewards = []
    actual_rewards = []
    episode_states  = []
    episode_actions = []

    while terminated == False:
        obs = obs.to(device)
        obs = obs.unsqueeze(0)

        obs_feat = encoder(obs)
        if action is None:
            current_state = rssm_model.init_state(obs_feat)
        else:
            current_state = rssm_model.observe_step(obs_feat, current_state['h'], current_state['s'], action.unsqueeze(0))

        action = planner(rssm_model, reward_model, current_state, device = device)
        y, r, d, t,_ =  env.step(action.cpu().numpy())
        decoded_pred = decoder(torch.cat([current_state['h'], current_state['s']], dim = -1)).cpu()
        frames.append((obs[0],decoded_pred[0]))
        pred_reward = reward_model(torch.cat([current_state['h'], current_state['s']], dim = -1)).squeeze().cpu().item()
        predicted_rewards.append(pred_reward)
        actual_rewards.append(r)
        episode_states.append(obs[0].cpu())
        episode_actions.append(action.cpu())
        terminated = d | t
        obs = y
    
    episode_states.append(obs)
    
    

    observations_t = torch.stack(episode_states, dim = 0)
    actions_t = torch.stack(episode_actions, dim = 0)
    rewards_t = torch.tensor(actual_rewards, dtype = torch.float32)



    episode = {
        "observations": observations_t,
        "actions": actions_t,
        "rewards": rewards_t
    }
    visu = {
        'frames': frames,
        'pred_rewards': pred_reward,
        'actual_rewards': actual_rewards
    }

    return episode, visu


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

    optimizer = torch.optim.Adam(
    list(rssm_model.parameters()) +
    list(reward_model.parameters()) +
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=cfg.lr
)

    populate_random(env, buffer, num_episodes = cfg.S)

    episode_rewards = deque(maxlen = 100)
    writer = SummaryWriter(log_dir="runs/planet_pendulum")
    save_path = "weights/"
    os.makedirs(save_path, exist_ok= True)

    for step in tqdm(range(cfg.num_train_step), desc= f'Step:'):
        losses, ep_reward = train(
            rssm_model,
            reward_model,
            cfg.C,
            cfg.R,
            cfg.T,
            cfg.L,
            cfg.eps,
            cfg.batch_size,
            buffer,
            encoder,
            decoder,
            optimizer,
            env,
            device
        )
        episode_rewards.append(ep_reward)

        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            value = value
            writer.add_scalar(f"Loss/{key}", value, step)
        
        writer.add_scalar(f"Reward/mean_reward",np.mean(episode_rewards), step)
        if step % 5 == 0:
            print(f"Mean reward: {np.mean(episode_rewards)}")
            save_model(save_path, rssm_model, reward_model, encoder, decoder)
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