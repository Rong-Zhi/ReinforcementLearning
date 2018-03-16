# Master thesis -- Deep Reinforcement Learning under Uncertainty in Autonomous Driving
**Supervisor from IAS lab: Dr. Joni Pajarinen**

Deep reinforcement learning has made huge progress in recent years with the help of many breakthrough technologies. Computers could automatically learn to play Atari games at human level from raw pixels, AlphaGo could even beat the world champions at Go. Partial observability is important in real world problems, for example, in real-time strategy games, the player usually could only observe partial environments of the game, where the rest is unknown to the player. In Autonomous driving, the agent car's sensors give a lot of information but e.g. cars behind corners, pedestrians behind cars cannot be observed using current sensors ....
POMDP is the standard model for such cases
which is the so-called POMDP problem. In autonomous driving, the agent car receives signals from noisy sensors and the envrionment of the real world is always partial observable, to address such problems, we will implement different algorithms, try to improve one of them or propose our own algorithm on CARLA.



## Application: CARLA
(application)

[CARLA](http://www.carla.org/) is an open-source simulator for autonomous research, supporting development, training, and validation of autonomous urban driving systems. In addition to open-source code and protocols, CARLA provides open digital assets (urban layouts, buildings, vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites and environmental conditions.

-- **15.03.2018 updates: CARLA has published [0.8.0](https://github.com/carla-simulator/carla/tree/release_0.8.0), Create their own pedestrian models, Remove Automotive Materials dependencies, New cheap background mountains generator**

We can build a POMDP environment by changing the weather conditions, setting the position of vehicles, or changing the behaviour of pedestrians on CARLA. Then run the simulator via python client for controlling the vehicle and saving data to disk.
(test algorithm on benchmark problems)

## Approaches for representing POMDP policies:
(Motivation..)

(Model-Free algorithms...)

(Guided reinforcement learning)

### General algorithms in RL
Policy gradients -- [Monte Carlo Policy Gradient(REINFORCE)](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/REINFORCE(VPG)) with Continuous Montain Car environment from Python scratch. Using Gaussian policy with RBF kernel (without baseline).

Policy gradients -- [Actor Critic method](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/Actor_Critic) with Continuous Montain Car environment by tensorflow. Using Gaussian policy with RBF kernel(with baselien).

[Relative Entropy Policy search](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/REPS) -- Policy search algorithm, implemented in Nchain problem of OPEN AI gym.

[Trust Region Policy Optmization](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/TRPO)- Build simple MLP network  

[Proximal Policy Optimization](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/PPO) -- build simple MLP network, tested in continous space -- continuous mountaincar environment and pendulum environment, and discrete POMDPS environment -- FVRS (Field Version Rock Sample), better performance compared with TRPO, worse than COPOS

### COPOS

### Guided Reinforcement learning

## Timeline:
Start Date: TBD
End Date: TBD

Check [timeplan](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/blob/master/Timeplan.pdf) for detailed information.

## Equipment
CARLA is built on a PC in IAS lab, with Ubuntu 17.10 system, 16GB memory, Intel core i5-8400 CPU (2.8GHz*6), GeForce GTX 1060 (3GB) graphic card. The following training/testing processes will also be implemented on this PC.
 
## References
- Reinforcement Learning [Tutorial](https://github.com/dennybritz/reinforcement-learning) given by Denny Britz
- [Deep Reinforcement learning for POMDPs](http://www.ausy.tu-darmstadt.de/uploads/Team/JoniPajarinen/master_thesis_hong_linh_thai_2018.pdf) master thesis of Thai, H. L.
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) from Schulman et al.
- [Relative Entropy Policy Search] [paper](https://pdfs.semanticscholar.org/ff47/526838ce85d77a50197a0c5f6ee5095156aa.pdf) from Peters et al.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) from Schulman et al.
