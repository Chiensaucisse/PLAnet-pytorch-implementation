import gymnasium as gym
import cv2
import torch
import numpy as np 


def to_tensor_obs(image: np.ndarray, size: tuple):
    

    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    return image





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
    



if __name__ == "__main__":

    env = TorchImageEnv('Pendulum-v1')

    x  = env.reset()

    print(x, x.shape)