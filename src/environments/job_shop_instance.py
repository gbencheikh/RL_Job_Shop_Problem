"""
Ce module définit la structure de données pour une instance de Job Shop Scheduling Problem.

Concepts clés :
- Job : Un ordre de fabrication est composé de plusieurs opérations séquentielles
- Opération : Une tâche qui doit être exécutée sur une machine spécifique
- Machine : Une ressource qui peut exécuter une opération à la fois
"""

from typing import List, Tuple
import random


class JobShopInstance:
    """
    Représente une instance du Job Shop Scheduling Problem.
    
    Attributes:
        num_jobs (int): Nombre de jobs dans l'instance
        num_machines (int): Nombre de machines disponibles
        jobs (List[List[Tuple[int, int]]]): 
            Liste des jobs, où chaque job est une liste d'opérations.
            Chaque opération est un tuple (machine_id, durée)
            
    Exemple:
        Instance 2x2 (2 jobs, 2 machines)
        jobs = [
            [(0, 3), (1, 2)],  # Job 0: Machine 0 pendant 3h, puis Machine 1 pendant 2h
            [(1, 2), (0, 4)]   # Job 1: Machine 1 pendant 2h, puis Machine 0 pendant 4h
        ]
    """
    
    def __init__(self, jobs: List[List[Tuple[int, int]]]):
        """
        Initialise une instance Job Shop.
        
        Args:
            jobs: Liste des jobs avec leurs opérations (machine, durée)
        """
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = self._compute_num_machines()
        self.total_operations = sum(len(job) for job in jobs)
        
    def _compute_num_machines(self) -> int:
        """Calcule le nombre de machines à partir des jobs."""
        machines = set()
        for job in self.jobs:
            for machine, _ in job:
                machines.add(machine)
        return len(machines)
    
    def get_operation(self, job_id: int, op_index: int) -> Tuple[int, int]:
        """
        Récupère une opération spécifique.
        
        Args:
            job_id: Identifiant du job (0 à num_jobs-1)
            op_index: Index de l'opération dans le job
            
        Returns:
            Tuple (machine_id, durée)
        """
        return self.jobs[job_id][op_index]
    
    def get_num_operations(self, job_id: int) -> int:
        """Retourne le nombre d'opérations pour un job donné."""
        return len(self.jobs[job_id])
    
    def __str__(self) -> str:
        """Représentation textuelle de l'instance."""
        result = "-" * 50 + "\n"
        result += f"Job Shop Instance: {self.num_jobs} jobs, {self.num_machines} machines\n"
        result += "-" * 50 + "\n"
        for job_id, job in enumerate(self.jobs):
            result += f"Job {job_id}: "
            operations = [f"M{machine}({duration})" for machine, duration in job]
            result += " → ".join(operations) + "\n"
        result += "-" * 50 + "\n"
        return result
    
    @staticmethod
    def create_simple_instance() -> 'JobShopInstance':
        """
        Crée une instance simple 2x2 pour tests.
        
        Returns:
            Instance avec 2 jobs et 2 machines
            
        Exemple de solution optimale:
            Machine 0: [Job0-Op0: 0-3] [Job1-Op1: 4-8]
            Machine 1: [Job1-Op0: 0-2] [Job0-Op1: 3-5]
            Makespan optimal = 8
        """
        jobs = [
            [(0, 3), (1, 2)],  # Job 0
            [(1, 2), (0, 4)]   # Job 1
        ]
        return JobShopInstance(jobs)
    
    @staticmethod
    def create_random_instance(num_jobs: int, num_machines: int, 
                              min_duration: int = 1, 
                              max_duration: int = 10) -> 'JobShopInstance':
        """
        Génère une instance aléatoire.
        
        Args:
            num_jobs: Nombre de jobs
            num_machines: Nombre de machines
            min_duration: Durée minimale d'une opération
            max_duration: Durée maximale d'une opération
            
        Returns:
            Instance aléatoire où chaque job passe par toutes les machines
        """
        jobs = []
        for _ in range(num_jobs):
            # Créer une permutation aléatoire des machines
            machine_order = list(range(num_machines))
            random.shuffle(machine_order)
            
            # Créer les opérations avec des durées aléatoires
            job = [(machine, random.randint(min_duration, max_duration)) 
                   for machine in machine_order]
            jobs.append(job)
        
        return JobShopInstance(jobs)
    
    @staticmethod
    def load_from_file(filepath: str) -> 'JobShopInstance':
        """
        Charge une instance depuis un fichier texte.
        
        Format du fichier:
            Ligne 1: num_jobs num_machines
            Lignes suivantes: Pour chaque job, paires (machine durée)
            
        Exemple de fichier:
            2 2
            0 3 1 2
            1 2 0 4
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Première ligne: nombre de jobs et machines
        num_jobs, num_machines = map(int, lines[0].split())
        
        jobs = []
        for i in range(1, num_jobs + 1):
            tokens = list(map(int, lines[i].split()))
            # Les tokens sont: machine1 durée1 machine2 durée2 ...
            job = [(tokens[j], tokens[j+1]) for j in range(0, len(tokens), 2)]
            jobs.append(job)
        
        return JobShopInstance(jobs)


# ============================================
# TESTS ET DÉMONSTRATION
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("DÉMONSTRATION - Job Shop Instance")
    print("=" * 60)
    print()
    
    # Test 1: Instance simple
    print("Test 1: Instance simple (2 jobs × 2 machines)")
    print("-" * 60)
    instance = JobShopInstance.create_simple_instance()
    print(instance)
    print()
    
    # Test 2: Instance aléatoire
    print("Test 2: Instance aléatoire (3 jobs × 3 machines)")
    print("-" * 60)
    random.seed(42)  # Pour reproductibilité
    random_instance = JobShopInstance.create_random_instance(3, 3, 1, 5)
    print(random_instance)
    print()
    
    # Test 3: Accès aux opérations
    print("Test 3: Accès aux opérations")
    print("-" * 60)
    machine, duration = instance.get_operation(0, 0)
    print(f"Job 0, Opération 0: Machine {machine}, Durée {duration}")
    print(f"Job 0 a {instance.get_num_operations(0)} opérations")
    print()
    
    print("✅ Tous les tests réussis !")