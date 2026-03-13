# Mobile Robot Navigation and Avoidance using Deep RL

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-brightgreen)
![PyGame](https://img.shields.io/badge/PyGame-2.5-yellow)

This repository contains an implementation of Deep Reinforcement Learning algorithms (**DQN** and **PPO**) designed to solve a dynamic 2D robotic navigation problem. It was developed as a comprehensive, modular framework emphasizing Software Engineering best practices, separation of concerns, and reproducible ML research.

## Task Objective
The agent controls a robot that must navigate an unknown, dynamic 2D environment to reach a target coordinate. The environment features:
- **Continuous State Space**: Powered by a simulated 200-resolution 2D Lidar sensor (range restricted to 20% of the map size) and absolute goal relative-positioning.
- **Dynamic Entities**: Moving creatures, including one orbiting the goal and others following random predefined trajectories.
- **Static Obstacles**: Fixed walls and minimum of 3 static obstacles.
- **Terminal States**: Reaching the goal (Success) or colliding with any entity/wall (Failure).