"""
Job shop instance representation.

Chaque instance contient:
- n_machines : int
- jobs : List[List[Tuple[int, float]]]
  where each job is a sequence of (machine_id, processing_time)

On fournit aussi un générateur d'instances de benchmark simple.
"""

from typing import List, Tuple
import random

Job = List[Tuple[int, float]]

class JobShopInstance:
    def __init__(self, jobs: List[Job], n_machines: int, name: str = "instance"):
        self.jobs = jobs
        self.n_machines = n_machines
        self.name = name

    def num_jobs(self) -> int:
        return len(self.jobs)

    def num_operations(self) -> int:
        return sum(len(j) for j in self.jobs)

    def __repr__(self):
        return f"<JobShopInstance {self.name} jobs={len(self.jobs)} machines={self.n_machines}>"

def generate_random_instance(n_jobs: int, n_machines: int, max_ops_per_job: int = 5, max_duration: int = 10, seed: int = None) -> JobShopInstance:
    """
    Génère une instance aléatoire.

    Params:
    - n_jobs, n_machines : tailles
    - max_ops_per_job : nombre maximal d'opérations par job
    - max_duration : durée max d'une opération
    - seed : pour reproductibilité
    """
    if seed is not None:
        random.seed(seed)

    jobs: List[Job] = []
    for j in range(n_jobs):
        n_ops = random.randint(1, max_ops_per_job)
        ops = []
        for _ in range(n_ops):
            m = random.randint(0, n_machines - 1)
            d = random.randint(1, max_duration)
            ops.append((m, d))
        jobs.append(ops)
    return JobShopInstance(jobs=jobs, n_machines=n_machines, name=f"rand_{n_jobs}x{n_machines}")
