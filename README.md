# ReinforcementLearning

This is my Reinforcement Learning Notes, Exercises, as well as Summary. Some images can't be shown on github, but it is possible to visit them from iPython Notebook.

E1 - E6, E8 Originally given by dennybritz: https://github.com/dennybritz/reinforcement-learning

E1: Get to know OPEN AI Gym 

E2: MDP and Model based method -- Dynammic Programming (Policy Iteration and Value Iteration)

E3: Model Free -- Monte Carlo 
    Implemented in BlackJack game - First-visit on-policy Method & Every-visit off-policy Method

E4: Model Free -- TD-Learning
    On-policy method -- SARSA , implemented in windy gridworld playground
    Off-policy method -- Q-Learning, implemented in Cliff playground
    
E5: Value Approximation -- use sklearn SGDregressor to train and update parameters for Q-Learning


E6: Deep Q Learning -- the CNN net we use here is published by DeepMind, train with Atari game (Breakout environment) images, target value = ground truth, Q value = predicted value, use RMSPropOptimizer to optimize parameters, code is given in Tensorflow, including state processor(resize Atari game images and transfer them into grayscale), checkpoint & monitor & model

E7: Policy gradients -- Monte Carlo Policy Gradient with Continuous Montain Car environment from Python scratch. Using Gaussian policy with RBF kernel (without baseline), there are some bugs in this code(can't update variance of gaussian policy)... check it latter

E8: Policy gradients -- Actor Critic method with Continuous Montain Car environment by tensorflow. Using Gaussian policy with RBF kernel(with baselien), still some bugs(large variance in the end). I guess it is the problem of the environment, the original code given by denny britz doesn't work properly either.... 

some tips for policy gradient: normalize the states into zero mean and unit variance and transform them into RBF features. Using advantage function(td error) to update both estimators instead of td-target. Try the algorithm in other environment

E9: Relative Entropy Policy search -- Policy search algorithm based on the paper from Jan Peters: https://pdfs.semanticscholar.org/ff47/526838ce85d77a50197a0c5f6ee5095156aa.pdf implemented in Nchain problem of OPEN AI gym.

E10: Trust Region Policy Optmization & Proximal Policy Optimization -- build simple MLP for both algorithms, test in continous mountaincar environment and pendulum environment.
