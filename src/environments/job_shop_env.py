# src/environments/job_shop_env.py
"""
Environnement Gymnasium pour un Job Shop simplifié.
Action : choix d'un job (index) à exécuter maintenant (si possible).
Observation : vecteur contenant l'état courant (indices d'opération, disponibilités machines, temps actuel).
Récompense : -makespan (ou -1 par pas) ; ici on donne une récompense intermédiaire
pour encourager ordre plus court: negative sum of remaining times normalized.

Note: c'est un environnement d'apprentissage initial, simple à comprendre et à améliorer.
"""

from typing import Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .job_shop_instance import JobShopInstance

class JobShopEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, instance: JobShopInstance, max_steps: int = 1000):
        super().__init__()
        self.instance = instance
        self.n_jobs = instance.num_jobs()
        self.n_machines = instance.n_machines
        self.max_steps = max_steps

        # State:
        # - next_op_index per job (int)
        # - machine_available_time per machine (float)
        # - current_time scalar
        # We'll normalize ranges loosely.
        obs_len = self.n_jobs + self.n_machines + 1
        high = np.inf * np.ones(obs_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Action: choose job index to schedule next (0 .. n_jobs-1)
        self.action_space = spaces.Discrete(self.n_jobs)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # per-job next operation index
        self.next_op = [0 for _ in range(self.n_jobs)]
        # machine availability times
        self.machine_ready = [0.0 for _ in range(self.n_machines)]
        self.current_time = 0.0
        self.steps = 0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        next_op_arr = np.array(self.next_op, dtype=np.float32)
        machine_arr = np.array(self.machine_ready, dtype=np.float32)
        cur = np.array([self.current_time], dtype=np.float32)
        obs = np.concatenate([next_op_arr, machine_arr, cur]).astype(np.float32)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        action: job index to schedule next.
        If chosen job has no remaining ops -> invalid -> small penalty.
        Otherwise we schedule its next op at max(current_time, machine_ready) and advance time.
        This is a simplified simulation that schedules one operation per step.
        """
        if self.done:
            return self._get_obs(), 0.0, True, True, {}

        self.steps += 1
        info = {}
        reward = 0.0

        # Validate action
        if action < 0 or action >= self.n_jobs:
            # invalid action
            reward = -1.0
            return self._get_obs(), reward, False, False, info

        job_idx = action
        job = self.instance.jobs[job_idx]
        op_idx = self.next_op[job_idx] if job_idx < len(self.next_op) else 0

        if op_idx >= len(job):
            # no-op left -> invalid
            reward = -0.5
            # small time advance to avoid stuck
            self.current_time += 0.1
            if self.steps >= self.max_steps:
                self.done = True
            return self._get_obs(), reward, self.done, False, info

        machine_id, duration = job[op_idx]
        start_time = max(self.current_time, self.machine_ready[machine_id])
        end_time = start_time + duration

        # schedule the operation
        self.machine_ready[machine_id] = end_time
        self.next_op[job_idx] += 1
        # Advance current time to the minimum next available machine or keep as end_time
        # Here we use greedy: move to earliest next finish
        self.current_time = min(self.machine_ready)  # approximate time progression

        # reward shaping: negative sum remaining processing times (encourage reducing)
        rem = 0.0
        for j_idx, j in enumerate(self.instance.jobs):
            for k in range(self.next_op[j_idx], len(j)):
                rem += j[k][1]
        reward = -rem / (1.0 + self.instance.num_operations())

        # terminal when all jobs finished
        finished = all(self.next_op[j] >= len(self.instance.jobs[j]) for j in range(self.n_jobs))
        if finished or self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), float(reward), self.done, False, info

    def render(self, mode="human"):
        print(f"t={self.current_time:.2f} next_op={self.next_op} machine_ready={self.machine_ready}")

    def close(self):
        pass
