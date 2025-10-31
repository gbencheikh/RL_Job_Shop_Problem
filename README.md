# Job Shop Scheduling avec Reinforcement Learning

## 📋 Description

Ce projet implémente un système de Reinforcement Learning pour résoudre le **Job Shop Scheduling Problem (JSSP)** en utilisant des algorithmes comme DQN et PPO.

### Qu'est-ce que le Job Shop Scheduling ?

Le Job Shop Scheduling est un problème d'optimisation combinatoire où :
- **n jobs** doivent être traités
- Chaque job a **m opérations** à effectuer dans un ordre spécifique
- Chaque opération nécessite une **machine particulière** pendant un certain temps
- Chaque machine ne peut traiter qu'**une opération à la fois**
- **Objectif** : Minimiser le temps total (makespan) pour compléter tous les jobs

### Exemple Simple (2 jobs × 2 machines)
```
Job 1: M1(3h) → M2(2h)
Job 2: M2(2h) → M1(4h)

Solution optimale : Makespan = 7h
```

## 🎯 Objectifs du Projet

1. ✅ Créer un environnement Gymnasium compatible pour Job Shop
2. ✅ Implémenter des agents RL (DQN, PPO)
3. ✅ Visualiser les solutions avec des diagrammes de Gantt
4. ✅ Tester sur des benchmarks classiques (FT06, FT10, etc.)
5. ✅ Comparer avec des heuristiques classiques

## 🛠️ Technologies

- **Python 3.12**
- **PyTorch** - Deep Learning
- **Gymnasium** - Environnement RL
- **Stable-Baselines3** - Algorithmes RL state-of-the-art
- **Matplotlib/Plotly** - Visualisation

## 📁 Structure du Projet
```
job-shop-rl/
├── src/
│   ├── environment/       # Environnement Job Shop
│   ├── agents/           # Agents RL (DQN, PPO)
│   ├── utils/            # Visualisation, logging
│   └── models/           # Architectures de réseaux
├── examples/             # Scripts d'entraînement
├── data/instances/       # Instances de test
└── results/              # Modèles et résultats
```

## 🚀 Installation
```bash
# Cloner le repository
git clone <votre-repo>
cd job-shop-rl

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Utilisation

### Entraîner un agent sur une instance simple
```bash
python examples/train_simple.py
```

### Entraîner sur des benchmarks
```bash
python examples/train_benchmark.py --instance FT06
```

### Évaluer un modèle
```bash
python examples/evaluate.py --model results/models/best_model.pth
```

## 📈 Résultats Attendus

- Graphiques de convergence de l'apprentissage
- Diagrammes de Gantt des solutions trouvées
- Comparaison avec heuristiques classiques (SPT, LPT, etc.)
- Temps de calcul et qualité des solutions

## 🎓 Concepts Clés - Reinforcement Learning

### État (State)
- Opérations déjà ordonnancées
- Machines disponibles
- Temps courant
- Opérations restantes

### Action
- Choisir la prochaine opération à ordonnancer

### Récompense
- Récompense négative = makespan (on veut minimiser)
- Bonus si on atteint un bon ordonnancement
- Pénalité si contraintes violées

## 📚 Références

- Fisher & Thompson (1963) - Instances FT
- Taillard (1993) - Instances benchmarks
- Sutton & Barto - Reinforcement Learning: An Introduction

## 👨‍💻 Auteur

[Votre Nom]

## 📝 License

MIT