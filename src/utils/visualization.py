"""
Visualization - Création de diagrammes de Gantt pour visualiser les solutions

Un diagramme de Gantt montre visuellement :
- Quelles opérations sont exécutées sur chaque machine
- Quand elles commencent et se terminent
- Les temps morts (idle time) des machines
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict
import numpy as np


class GanttVisualizer:
    """
    Créateur de diagrammes de Gantt pour le Job Shop.
    """
    
    def __init__(self):
        # Palette de couleurs pour les jobs
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
            '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
            '#F8B739', '#52B788', '#E07A5F', '#3D5A80'
        ]
    
    def plot_schedule(self, schedule: List[Dict], num_machines: int, 
                     title: str = "Job Shop Schedule - Diagramme de Gantt",
                     save_path: str = None):
        """
        Crée un diagramme de Gantt à partir d'un ordonnancement.
        
        Args:
            schedule: Liste de dictionnaires avec les opérations
                     Chaque dict contient: job_id, machine_id, start_time, end_time, duration
            num_machines: Nombre de machines
            title: Titre du graphique
            save_path: Chemin pour sauvegarder l'image (optionnel)
        """
        if not schedule:
            print("Ordonnancement vide, rien à visualiser")
            return
        
        # Calculer le makespan
        makespan = max(op['end_time'] for op in schedule)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(14, max(6, num_machines * 0.8)))
        
        # Pour chaque machine, tracer les opérations
        for machine_id in range(num_machines):
            # Récupérer les opérations de cette machine
            machine_ops = [op for op in schedule if op['machine_id'] == machine_id]
            machine_ops.sort(key=lambda x: x['start_time'])
            
            # Tracer chaque opération
            for op in machine_ops:
                job_id = op['job_id']
                start = op['start_time']
                duration = op['duration']
                
                # Couleur selon le job
                color = self.colors[job_id % len(self.colors)]
                
                # Créer un rectangle pour l'opération
                rect = mpatches.Rectangle(
                    (start, machine_id - 0.4),  # Position (x, y)
                    duration,                    # Largeur
                    0.8,                        # Hauteur
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Ajouter le texte (Job ID) au centre
                center_x = start + duration / 2
                center_y = machine_id
                ax.text(center_x, center_y, f'J{job_id}',
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='white')
        
        # Configuration des axes
        ax.set_xlim(0, makespan * 1.05)
        ax.set_ylim(-0.5, num_machines - 0.5)
        
        # Labels
        ax.set_xlabel('Temps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Machines', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\nMakespan = {makespan:.1f}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Ticks des machines
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
        
        # Grille
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Légende (jobs)
        num_jobs = max(op['job_id'] for op in schedule) + 1
        legend_patches = [
            mpatches.Patch(color=self.colors[i % len(self.colors)], 
                          label=f'Job {i}')
            for i in range(num_jobs)
        ]
        ax.legend(handles=legend_patches, loc='upper right', 
                 bbox_to_anchor=(1.12, 1), fontsize=10)
        
        plt.tight_layout()
        
        # Sauvegarder si demandé
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            saved(f"Graphique sauvegardé",save_path)
        
        plt.show()
    
    def compare_schedules(self, schedules: Dict[str, List[Dict]], 
                         num_machines: int,
                         save_path: str = None):
        """
        Compare plusieurs ordonnancements côte à côte.
        
        Args:
            schedules: Dictionnaire {nom: ordonnancement}
            num_machines: Nombre de machines
            save_path: Chemin pour sauvegarder
        """
        num_schedules = len(schedules)
        fig, axes = plt.subplots(1, num_schedules, 
                                figsize=(7 * num_schedules, max(6, num_machines * 0.8)))
        
        if num_schedules == 1:
            axes = [axes]
        
        for idx, (name, schedule) in enumerate(schedules.items()):
            ax = axes[idx]
            makespan = max(op['end_time'] for op in schedule) if schedule else 0
            
            # Tracer les opérations
            for machine_id in range(num_machines):
                machine_ops = [op for op in schedule if op['machine_id'] == machine_id]
                machine_ops.sort(key=lambda x: x['start_time'])
                
                for op in machine_ops:
                    job_id = op['job_id']
                    start = op['start_time']
                    duration = op['duration']
                    color = self.colors[job_id % len(self.colors)]
                    
                    rect = mpatches.Rectangle(
                        (start, machine_id - 0.4),
                        duration, 0.8,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5,
                        alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    center_x = start + duration / 2
                    center_y = machine_id
                    ax.text(center_x, center_y, f'J{job_id}',
                           ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color='white')
            
            # Configuration
            ax.set_xlim(0, makespan * 1.05)
            ax.set_ylim(-0.5, num_machines - 0.5)
            ax.set_xlabel('Temps', fontsize=11, fontweight='bold')
            ax.set_ylabel('Machines', fontsize=11, fontweight='bold')
            ax.set_title(f'{name}\nMakespan = {makespan:.1f}', 
                        fontsize=12, fontweight='bold')
            ax.set_yticks(range(num_machines))
            ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            saved(f"Comparaison sauvegardée", save_path)
        
        plt.show()


def plot_training_curve(rewards: List[float], 
                       window_size: int = 100,
                       title: str = "Courbe d'Apprentissage",
                       save_path: str = None):
    """
    Trace la courbe d'apprentissage (récompenses au fil du temps).
    
    Args:
        rewards: Liste des récompenses (une par épisode)
        window_size: Taille de la fenêtre pour la moyenne mobile
        title: Titre du graphique
        save_path: Chemin pour sauvegarder
    """
    plt.figure(figsize=(12, 6))
    
    episodes = list(range(len(rewards)))
    
    # Tracer les récompenses brutes
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Récompense brute')
    
    # Calculer et tracer la moyenne mobile
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moyenne mobile ({window_size} ép.)')
    
    plt.xlabel('Épisode', fontsize=12, fontweight='bold')
    plt.ylabel('Récompense (= -Makespan)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved(f"Courbe sauvegardée",save_path)
    
    plt.show()


# ============================================
# TESTS ET DÉMONSTRATION
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("DÉMONSTRATION - Visualisation Gantt")
    print("=" * 60)
    print()
    
    # Créer un exemple d'ordonnancement
    schedule_example = [
        {'job_id': 0, 'op_index': 0, 'machine_id': 0, 'start_time': 0, 'duration': 3, 'end_time': 3},
        {'job_id': 0, 'op_index': 1, 'machine_id': 1, 'start_time': 3, 'duration': 2, 'end_time': 5},
        {'job_id': 1, 'op_index': 0, 'machine_id': 1, 'start_time': 0, 'duration': 2, 'end_time': 2},
        {'job_id': 1, 'op_index': 1, 'machine_id': 0, 'start_time': 3, 'duration': 4, 'end_time': 7},
    ]
    
    visualizer = GanttVisualizer()
    
    print("Création du diagramme de Gantt...")
    visualizer.plot_schedule(
        schedule_example, 
        num_machines=2,
        title="Exemple Simple - Instance 2×2"
    )
    
    print("\n✅ Visualisation terminée !")