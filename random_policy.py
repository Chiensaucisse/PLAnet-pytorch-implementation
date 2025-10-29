import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np


def run_random_policy(env, num_episodes: int = 10, max_steps: int = 500):


    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            env.render()

            # Random action
            action = env.action_space.sample()

        
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} — Reward: {total_reward:.2f}")

    env.close()

    return episode_rewards


if __name__ == "__main__":
    env = gym.make("Pendulum-v1", render_mode="human")
    print(env.action_space)
    print(env.observation_space)

    num_episodes = 10
    max_steps = 500

    episode_rewards = run_random_policy(env= env, num_episodes= num_episodes, max_steps= max_steps)

    plt.figure(figsize=(7,4))
    plt.plot(episode_rewards, marker='o')
    plt.title("Random Policy Performance on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()



