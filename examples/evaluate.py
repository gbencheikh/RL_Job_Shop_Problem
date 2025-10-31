# examples/evaluate.py
"""
Évaluer un agent enregistré sur une instance.
"""

import torch
from src.environments.job_shop_instance import generate_random_instance
from src.environments.job_shop_env import JobShopEnv
from src.agents.DQN_agent import DQNAgent

def evaluate(model_path="results/models/dqn_simple.pth", n_runs=5):
    instance = generate_random_instance(n_jobs=4, n_machines=3, seed=1)
    env = JobShopEnv(instance)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim, device="cpu")
    agent.load(model_path)

    for i in range(n_runs):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.act(obs)
            obs, r, done, _, _ = env.step(a)
            total += r
        print(f"Run {i} total reward {total:.3f}")

if __name__ == "__main__":
    evaluate()
