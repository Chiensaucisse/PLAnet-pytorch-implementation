import gymnasium as gym
import torch
from torch import nn 
import numpy
from memory import ReplayBuffer
from model import RSSM, RewardModel, ConvEncoder, ConvDecoder
from tqdm import tqdm
from utils import *



def populate_random(env: gym.Env, buffer: ReplayBuffer,  num_episodes: int = 10):


    

    for episode in tqdm(range(num_episodes), desc="Populating with random policy"):
        state, _ = env.reset()
        terminated  = False
        observations_list = []
        actions_list = []
        rewards_list = []
        while terminated == False:

            # Random action
            action = env.action_space.sample()

        
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
    ) -> None:


    observations_image = batch['observations']
    actions =  batch['actions']
    rewards = batch['rewards']

    obs_feat = encoder(observations_image)

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
          batch_size: int,
          buffer: ReplayBuffer,
          encoder: nn.Module,
          optim,
          device
           ):


    for _ in range():
        episode_states  = []
        episode_actions = []
        episodes_rewards = []


        batch = buffer.sample(batch_size)
        # TO DO
        fit_rssm(rssm_model,
                reward_model,
                encoder,
                batch

                )
        
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

def main():

    env = gym.make("Pendulum-v1", render_mode="human")
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    action_size = env.action_space.shape[0]
    max_capacity = 500
    buffer = ReplayBuffer(capacity=max_capacity)

    rssm_model = RSSM(action_size= action_size)

    total_reward = 0.0
    min_reward = 100

    populate_random(env, buffer, num_episodes = 100)


    while total_reward <= min_reward:

        train()



    return 




if __name__ == "__main__":
    main()