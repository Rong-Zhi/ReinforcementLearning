# My Master thesis -- Deep Reinforcement Learning under Uncertainty in Autonomous Driving

Nowadays, deep reinforcement learning is 

E1: Get to know [OPEN AI Gym](https://github.com/Rong-Zhi/ReinforcementLearning/code/openai_gym) 

E2: MDP and Model based method -- [Dynammic Programming](https://github.com/Rong-Zhi/ReinforcementLearning/code/Dynamic_Programming) (Policy Iteration and Value Iteration)

E3: Model Free -- [Monte Carlo](https://github.com/Rong-Zhi/ReinforcementLearning/code/Model_Free) 
    Implemented in BlackJack game - First-visit on-policy Method & Every-visit off-policy Method

E4: Model Free -- [TD-Learning](https://github.com/Rong-Zhi/ReinforcementLearning/code/Model_Free)
    On-policy method -- SARSA , implemented in windy gridworld playground
    Off-policy method -- Q-Learning, implemented in Cliff playground
    
E5: [Value Approximation](https://github.com/Rong-Zhi/ReinforcementLearning/code/Value_Approximation) -- use sklearn SGDregressor to train and update parameters for Q-Learning


E6: [Deep Q Learning](https://github.com/Rong-Zhi/ReinforcementLearning/code/Deep_Q_Learning) -- the CNN net we use here is published by DeepMind, train with Atari game (Breakout environment) images, target value = ground truth, Q value = predicted value, use RMSPropOptimizer to optimize parameters, code is given in Tensorflow, including state processor(resize Atari game images and transfer them into grayscale), checkpoint & monitor & model

E7: Policy gradients -- [Monte Carlo Policy Gradient(REINFORCE)](https://github.com/Rong-Zhi/ReinforcementLearning/code/REINFORCE) with Continuous Montain Car environment from Python scratch. Using Gaussian policy with RBF kernel (without baseline), there are some bugs in this code(can't update variance of gaussian policy)... check it latter

E8: Policy gradients -- [Actor Critic method](https://github.com/Rong-Zhi/ReinforcementLearning/code/Actor_Critic) with Continuous Montain Car environment by tensorflow. Using Gaussian policy with RBF kernel(with baselien), still some bugs(large variance in the end). I guess it is the problem of the environment, the original code given by denny britz doesn't work properly either.... 

some tips for policy gradient: normalize the states into zero mean and unit variance and transform them into RBF features. Using advantage function(td error) to update both estimators instead of td-target. Try the algorithm in other environment

E9: [Relative Entropy Policy search](https://github.com/Rong-Zhi/ReinforcementLearning/code/REPS) -- Policy search algorithm, implemented in Nchain problem of OPEN AI gym.

E10: [Trust Region Policy Optmization](https://github.com/Rong-Zhi/ReinforcementLearning/code/TRPO)- Build simple MLP network  
E10: [Proximal Policy Optimization](https://github.com/Rong-Zhi/ReinforcementLearning/code/PPO) -- build simple MLP network, first tested in continous space -- continuous mountaincar environment and pendulum environment, then tested in discrete POMDPS environment -- FVRS (Field Version Rock Sample)


## References
- Reinforcement Learning [Tutorial](https://github.com/dennybritz/reinforcement-learning) given by Denny Britz
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) from Schulman et al.
- [Relative Entropy Policy Search] [paper](https://pdfs.semanticscholar.org/ff47/526838ce85d77a50197a0c5f6ee5095156aa.pdf) from Peters et al.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) from Schulman et al.
