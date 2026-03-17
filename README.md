# Mobile Robot Navigation and Avoidance using Deep RL

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-brightgreen)
![PyGame](https://img.shields.io/badge/PyGame-2.5-yellow)

This repository contains an implementation of Deep Reinforcement Learning algorithms (**DQN** and **PPO**) designed to solve a dynamic 2D robotic navigation problem. It was developed as a comprehensive, modular framework emphasizing Software Engineering best practices, separation of concerns, and reproducible ML research.



## 📽️ Agent Behavior
| Agent | Navigation & Obstacle Avoidance Demo |
| :---: | :---: |
| **PPO** | ![alt text](output/videos/ppo_eval.gif) |
| **DQN** | ![alt text](output/videos/dqn_eval.gif) |

---

## 📂 Repository Structure
```text
.
├── assets/                 # Robot and environment sprites
├── docs/                   # Documentation (Report, Docstrings)
├── nav2d/                  # Core package
│   ├── agents/             # DQN and PPO implementations
│   ├── envs/               # Environment engine, sensors, and Gym wrapper
│   ├── utils/              # Logging and visualization utilities
│   └── config.py           # Configuration file
├── output/                 # Logs, model checkpoints, and generated videos
├── scripts/                # Entry points for training and evaluation
│   ├── train_dqn.py        # DQN training script
│   ├── train_ppo.py        # PPO training script
│   └── evaluate.py         # Evaluation utility
└── requirements.txt        # Project dependencies
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies in a virtual environment (Recommended):
```bash
pip install -r requirements.txt
```

### 2. Training
To train an agent, run the corresponding script in the `scripts/` directory.

**Train PPO (Recommended):**
```bash
python scripts/train_ppo.py
```

**Train DQN:**
```bash
python scripts/train_dqn.py
```

### 3. Evaluation
Use the evaluation script to test a saved model. This will output performance metrics (Success/Collision rates) and can optionally record the agent's behavior.

```bash
python scripts/evaluate.py --algo <dqn/ppo> \
                           --model_path output/models/<model_name>.pth \
                           --episodes 100 \
                           --record
```

**Available Flags:**
- `--algo`: Algorithm to use (`dqn` or `ppo`).
- `--model_path`: Path to the `.pth` weight file.
- `--episodes`: Number of evaluation trials (default: 100).
- `--record`: Flag to generate an MP4 video of the evaluation.

---

## 🛠️ Configuration
Environmental and hyperparameter settings are centralized in `nav2d/config.py`. Adjust these parameters to fine-tune the reward shaping, Lidar resolution, or learning hyperparameters.

---