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


    def get_action(self, observation):
        if rnd.random() < self.epsilon:
            return np.max(self.q_values[observation])
        else:
            return np.array([0.0, 1.0, 0.0])
        
    def update(self, curr_state, action, reward, next_state):
        best_next_q = np.max(self.q_values[next_state])
        self.q_values[curr_state][action] = (1 - self.learning_rate) * self.q_values[curr_state][action] + self.learning_rate * (reward + self.discount_factor * best_next_q)


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
                    continuous=config["environment"]["action_type"])
    agent = CarRacingAgent(config, env)

    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=42)

    env.close()



if __name__ == "__main__":
    main()  