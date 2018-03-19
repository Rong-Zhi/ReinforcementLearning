# Master thesis -- Deep Reinforcement Learning under Uncertainty in Autonomous Driving
**Supervisor from IAS lab: Dr. Joni Pajarinen**

Deep reinforcement learning has made huge progress in recent years with the help of many breakthrough technologies. Computers could automatically learn to play Atari games at human level from raw pixels, AlphaGo could even beat the world champions at Go. Partial observability is important in real world problems, for example, in real-time strategy games, the player usually could only observe partial environments of the game, where the rest is unknown to the player. In Autonomous driving, receives signals from noisy sensors but just partial information of the surrounding environment, e.g. cars behind corners, pedestrians behind cars cannot be observed using current sensors, POMDP (Partial Observable Markov Decision Process) is the standard model for such cases. In this thesis, we will implement different algorithms, try to improve one of them or propose our own algorithm to solve POMDP problems in autonomous driving, train and test the algortihms on an open-source autonomus driving simulator -- CARLA.


## CARLA simulator introduction and modification

[CARLA](http://www.carla.org/) is an open-source simulator for autonomous research, supporting development, training, and validation of autonomous urban driving systems. In addition to open-source code and protocols, CARLA provides open digital assets (urban layouts, buildings, vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites and environmental conditions.

-- **15.03.2018 updates: CARLA has published [0.8.0](https://github.com/carla-simulator/carla/tree/release_0.8.0), Create their own pedestrian models, Remove Automotive Materials dependencies, New cheap background mountains generator**

We can build a POMDP environment by changing the weather conditions, setting the position of vehicles, or changing the behaviour of pedestrians on CARLA. Then run the simulator via python client for controlling the vehicle and saving data to disk. To avoid unknown issues we could face in CARLA, we want thus test algorithms on other reinforcement learning benchmark tasks first, e.g.OpenAI Gym.

## Algorithms to solve POMDPs:
As we described before, in POMDP problems, the agent does not directly observe the environment's state, and must make decisions under uncertainty of the true environment state. One method is model-free approach, which directly learn the policy by interactions with environment, and thus, without learning the model of the environment. An alternative solution could be model-based approach, which learns a POMDP model of the environment and afterwards compute an optimal policy based on the learned model. In this thesis, we will focus on model-free approaches.

### General algorithms in RL
Here we implemented some model-free RL algorithms, we will use some of them (TRPO and PPO) as baseline for POMDP problems.

Policy gradients -- [Monte Carlo Policy Gradient(REINFORCE)](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/REINFORCE(VPG)) with Continuous Montain Car environment from Python scratch. Using Gaussian policy with RBF kernel (without baseline).

Policy gradients -- [Actor Critic method](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/Actor_Critic) with Continuous Montain Car environment by tensorflow. Using Gaussian policy with RBF kernel(with baselien).

[Relative Entropy Policy search](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/REPS) -- Policy search algorithm, implemented in Nchain problem of OPEN AI gym.

[Trust Region Policy Optmization](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/TRPO)- Build simple MLP network  

[Proximal Policy Optimization](https://git.ias.informatik.tu-darmstadt.de/zhi/ReinforcementLearning/tree/master/code/PPO) -- build simple MLP network, tested in continous space -- continuous mountaincar environment and pendulum environment, and discrete POMDPS environment -- FVRS (Field Version Rock Sample), better performance compared with TRPO, worse than COPOS

### COPOS
Compatible policy search (COPOS)

### Guided Reinforcement learning
In POMDP autonomous driving environment, the agent car observes relevant environmental features by sensors, which shows some additional possible problems compared to MDP, for example, expensive computational cost due to too much observed data, the agent car has to operate even with insufficient sensor data, 
 Information gathering and exploitation. 
 
 With a guided reinforcement learning approach, one can first learn policy by combining also the real states, we call it guided policy, then train the policy with this guided policy to ultimately improve the performance in abovementioned problems.
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
- [Learning Deep Control Policies for Autonomous Aerial Vehicles with MPC-Guided Policy Search](http://rll.berkeley.edu/icra2016mpcgps/ICRA16_MPCGPS) from Zhang et al.
