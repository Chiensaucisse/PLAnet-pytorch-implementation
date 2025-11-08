import torch
from torch import nn
import numpy 
import collections
from queue import deque
import random




class ReplayBuffer():

    def __init__(self, capacity: int, device = 'cpu') -> None:

        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)
        self.device = device
    
    def __len__(self):
        return len(self.buffer)
    
    def add_episode(self, episode: dict):
        self.buffer.append(episode)

    def sample(self, batch_size: int, chunk_length: int) -> dict:
        
        episodes = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs_batch, rewards_batch, actions_batch = [], [], []

        for episode in episodes:
            
            observations  = episode['observations'] # T + 1
            rewards  = episode['rewards'] # T
            actions = episode['actions'] # T 

            T = len(actions)
            if  T < chunk_length:
                continue
            start = random.randint(0, T - chunk_length)
            end = start + chunk_length

            obs_seq = observations[start: end + 1] # L + 1
            act_seq = actions[start: end] # L
            rew_seq =  rewards[start : end]

            obs_batch.append(obs_seq)
            actions_batch.append(act_seq)
            rewards_batch.append(rew_seq)
        
        obs_batch = torch.stack(obs_batch, dim = 0).float().to(device=self.device)
        act_batch = torch.stack(actions_batch, dim = 0).float().to(device=self.device)
        rewards_batch = torch.stack(rewards_batch, dim = 0).float().to(device=self.device)
        
        return {
            "observations": obs_batch,
            "actions": act_batch,
            "rewards": rewards_batch
        }
    
    
    def populate(self, data: list[tuple]):

        observations_list, actions_list, rewards_list = zip(*data)

        observations_t = torch.stack(observations_list, dim = 0)
        actions_t = torch.stack(actions_list, dim = 0)
        rewards_t  =torch.stack(rewards_list, dim = 0)

        episode = {
            "observations": observations_t,
            "actions": actions_t,
            "rewards": rewards_t
        }

        self.add_episode(episode)