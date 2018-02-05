# ReinforcementLearning

This is my Reinforcement Learning Notes, Exercises, as well as Summary.

Originally given by dennybritz: https://github.com/dennybritz/reinforcement-learning


E1: Get to know OPen AI Gym 

E2: MDP and Model based method -- Dynammic Programming (Policy Iteration and Value Iteration)

E3: Model Free -- Monte Carlo 
    Implemented in BlackJack game - First-visit on-policy Method & Every-visit off-policy Method

E4: Model Free -- TD-Learning
    On-policy method -- SARSA , implemented in windy gridworld playground
    Off-policy method -- Q-Learning, implemented in Cliff playground
    
E5: Value Approximation -- use sklearn SGDregressor to train and update parameters for Q-Learning


E6: Deep Q Learning -- the CNN net we use here is published by DeepMind, train with Atari game (Breakout environment) images, target value = ground truth, Q value = predicted value, use RMSPropOptimizer to optimize parameters, code is given in Tensorflow, including state processor(resize Atari game images and transfer them into grayscale), checkpoint & monitor & model

E7: Policy gradients --
