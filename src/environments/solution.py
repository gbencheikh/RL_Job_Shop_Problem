"""
Solution - Représente et évalue une solution du Job Shop Problem

Une solution est un ordonnancement (schedule) qui spécifie:
- Quand chaque opération commence
- Sur quelle machine elle s'exécute
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Operation:
    """
    Représente une opération ordonnancée.
    
    Attributes:
        job_id: ID du job
        op_index: Index de l'opération dans le job
        machine_id: Machine sur laquelle l'opération s'exécute
        start_time: Temps de début
        duration: Durée de l'opération
    """
    job_id: int
    op_index: int
    machine_id: int
    start_time: int
    duration: int
    
    @property
    def end_time(self) -> int:
        """Temps de fin de l'opération."""
        return self.start_time + self.duration
    
    def __repr__(self) -> str:
        return f"J{self.job_id}-O{self.op_index}[M{self.machine_id}: {self.start_time}-{self.end_time}]"


class JobShopSolution:
    """
    Représente une solution complète au Job Shop Problem.
    """
    
    def __init__(self, instance):
        """
        Initialise une solution vide.
        
        Args:
            instance: L'instance JobShopInstance à résoudre
        """
        self.instance = instance
        self.schedule: List[Operation] = []
        self.makespan: Optional[int] = None
        
    def add_operation(self, job_id: int, op_index: int, start_time: int):
        """
        Ajoute une opération à l'ordonnancement.
        
        Args:
            job_id: ID du job
            op_index: Index de l'opération
            start_time: Temps de début
        """
        machine_id, duration = self.instance.get_operation(job_id, op_index)
        op = Operation(job_id, op_index, machine_id, start_time, duration)
        self.schedule.append(op)
        
    def compute_makespan(self) -> int:
        """
        Calcule le makespan (temps total) de la solution.
        
        Returns:
            Le makespan (temps de fin de la dernière opération)
        """
        if not self.schedule:
            return 0
        self.makespan = max(op.end_time for op in self.schedule)
        return self.makespan
    
    def is_feasible(self) -> Tuple[bool, str]:
        """
        Vérifie si la solution est réalisable (respecte toutes les contraintes).
        
        Returns:
            (True/False, message d'erreur si non réalisable)
        """
        # Vérifier que toutes les opérations sont présentes
        if len(self.schedule) != self.instance.total_operations:
            return False, f"Manque des opérations: {len(self.schedule)}/{self.instance.total_operations}"
        
        # Vérifier les précédences (ordre des opérations dans chaque job)
        job_last_end = {}
        for op in sorted(self.schedule, key=lambda x: (x.job_id, x.op_index)):
            if op.job_id in job_last_end:
                if op.start_time < job_last_end[op.job_id]:
                    return False, f"Violation de précédence pour Job {op.job_id}"
            job_last_end[op.job_id] = op.end_time
        
        # Vérifier qu'aucune machine n'a deux opérations en même temps
        for machine in range(self.instance.num_machines):
            machine_ops = [op for op in self.schedule if op.machine_id == machine]
            machine_ops.sort(key=lambda x: x.start_time)
            
            for i in range(len(machine_ops) - 1):
                if machine_ops[i].end_time > machine_ops[i+1].start_time:
                    return False, f"Conflit sur Machine {machine}: {machine_ops[i]} et {machine_ops[i+1]}"
        
        return True, "Solution réalisable ✅"
    
    def get_machine_schedule(self, machine_id: int) -> List[Operation]:
        """Retourne les opérations d'une machine, triées par temps de début."""
        ops = [op for op in self.schedule if op.machine_id == machine_id]
        return sorted(ops, key=lambda x: x.start_time)
    
    def __str__(self) -> str:
        """Représentation textuelle de la solution."""
        feasible, msg = self.is_feasible()
        result = f"Job Shop Solution\n"
        result += f"Makespan: {self.compute_makespan()}\n"
        result += f"Statut: {msg}\n"
        result += "-" * 60 + "\n"
        
        for machine in range(self.instance.num_machines):
            result += f"Machine {machine}: "
            ops = self.get_machine_schedule(machine)
            if ops:
                op_strs = [f"J{op.job_id}({op.start_time}-{op.end_time})" for op in ops]
                result += " ".join(op_strs)
            else:
                result += "(vide)"
            result += "\n"
        
        return result


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    from job_shop_instance import JobShopInstance
    
    print("=" * 60)
    print("DÉMONSTRATION - Job Shop Solution")
    print("=" * 60)
    print()
    
    # Créer une instance simple
    instance = JobShopInstance.create_simple_instance()
    print("Instance:")
    print(instance)
    
    # Créer une solution optimale
    print("Solution optimale:")
    print("-" * 60)
    solution = JobShopSolution(instance)
    
    # Job 0: Op0 sur M0 (0-3), Op1 sur M1 (3-5)
    solution.add_operation(0, 0, 0)  # J0-O0 commence à t=0
    solution.add_operation(0, 1, 3)  # J0-O1 commence à t=3
    
    # Job 1: Op0 sur M1 (0-2), Op1 sur M0 (4-8)
    solution.add_operation(1, 0, 0)  # J1-O0 commence à t=0
    solution.add_operation(1, 1, 4)  # J1-O1 commence à t=4
    
    print(solution)
    
    # Test d'une solution non réalisable
    print("\nTest: Solution avec conflit")
    print("-" * 60)
    bad_solution = JobShopSolution(instance)
    bad_solution.add_operation(0, 0, 0)
    bad_solution.add_operation(1, 1, 0)  # Conflit sur M0 !
    bad_solution.add_operation(1, 0, 0)
    bad_solution.add_operation(0, 1, 1)
    
    print(bad_solution)