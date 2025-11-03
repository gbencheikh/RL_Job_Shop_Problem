"""
Utilitaires pour créer et charger des instances de job-shop.
"""
from typing import List
from ..environments.job_shop_instance import generate_random_instance, JobShopInstance

def small_example() -> JobShopInstance:
    # Exemple très simple (3 jobs, 3 machines)
    jobs = [
        [(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)]
    ]
    return JobShopInstance(jobs=jobs, n_machines=3, name="small_example")

def random_benchmark(seed: int = 0):
    return generate_random_instance(n_jobs=5, n_machines=4, max_ops_per_job=4, max_duration=10, seed=seed)
