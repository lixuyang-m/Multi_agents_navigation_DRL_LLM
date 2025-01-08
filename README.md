# Enhancing Deep Reinforcement Learning via Large Language Model for Multi-Robot Navigation 

This repository heavily relies on [Xi Lin's work](https://github.com/RobustFieldAutonomyLab/Multi_Robot_Distributional_RL_Navigation). We use the Adaptive_IQN algorithm proposed in [Xi Lin's paper](https://arxiv.org/abs/2402.11799) as the benchmark algorithm. Currently, we are attempting to use large language models (LLM) to guide agents during the training process.

Based on the original code repository, we made modifications to the trainer.py file by loading the pre-trained LLM and providing guidance at critical moments. The guidance from the LLM was given in textual form and mapped to the action space’s probability distribution in a simple manner. Then, in the agents.py file, we weighted the LLM's guidance at the value function level using the IQN algorithm. Additionally, we have made ablation experiments, including:

(1) Designing separate experience replay buffers specifically for the LLM guidance.

(2) Introducing reward plasticity, along with additional punishment when the TTC = 0.25s triggers LLM guidance.

(3) During sampling, incorporating extra JS loss terms based on the experience guided by the LLM, with weight coefficients that decay over time.

<<<<<<< HEAD
In order to debug the guidance method of LLM, we first simplified the environment of the benchmark algorithm and removed the influence of vortices and water flow in the scene. To ensure fairness in the comparison, all other settings of the algorithm remain unchanged. The current experimental results are saved in the training.py file, and the evaluation visualization results during the training process are as follows:
=======
In order to debug the guidance method of LLM, we first simplified the environment of the benchmark algorithm and removed the influence of vortices and water flow in the scene. To ensure fairness in the comparison, all other settings of the algorithm remain unchanged. The current experimental results are saved in the training.py file, and the evaluation visualization results during the training process are as follows:


>>>>>>> ad435694628b0d98d9d9a7e0c415c505f4ae5e47