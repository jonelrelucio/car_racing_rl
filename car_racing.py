import gymnasium as gym
import numpy as np
import random as rnd
import math
import yaml
from collections import defaultdict


class CarRacingAgent:

    def __init__(self, config, env):
        self.env = env

        # Trained values
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.learning_rate = config["hyperparameters"]["learning_rate"]
        self.discount_factor = config["hyperparameters"]["discount_factor"]

        # Exploration parameters
        self.epsilon = config["hyperparameters"]["epsilon"]
        self.exploration_decay = config["hyperparameters"]["exploration_decay"]
        self.final_epsilon = config["hyperparameters"]["final_epsilon"]

        # logging
        self.training_error = []       


    def get_action(self, obs):
        if rnd.random() < (1 - self.epsilon):
            return np.argmax(self.q_values[obs])
        else:
            return self.env.action_space.sample()
        
    def update(self, curr_obs, action, reward, next_obs):
        future_q_value= np.max(self.q_values[next_obs])
        past_q_value = (1 - self.q_values[curr_obs][action])
        target = self.learning_rate * (reward + self.discount_factor * future_q_value)
        self.q_values[curr_obs][action] = past_q_value + target

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
    
def main():
    with open("config.yaml", "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print("Error parsing config.yaml:", exc)
            return

    env = gym.make(config["environment"]["name"],
                    render_mode=config["environment"]["render_mode"], 
                    lap_complete_percent=0.95, 
                    domain_randomize=False, 
                    continuous=False)
    agent = CarRacingAgent(config, env)
    n_episodes = config["environment"]["n_episodes"]

    for _ in range(n_episodes):
        observation, info = env.reset(seed=42)
        observation = totuple(observation)
        done = False
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = totuple(next_observation)

            agent.update(observation, action, reward, next_observation)
            observation = next_observation

            done = terminated or truncated

            done = terminated or truncated
        agent.epsilon *= agent.exploration_decay



    env.close()



if __name__ == "__main__":
    main()  