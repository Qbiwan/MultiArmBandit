import numpy as np


class MultiArmBandit:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.reward = np.random.randn(k)
        self.Q = None
        self.N = None
        self.initQ()
        self.check_epsilon(epsilon)

    def initQ(self):
        self.Q = np.array([0.0]*self.k)
        self.N = [0]*self.k

    def check_epsilon(self, epsilon):
        if 0.0 <= epsilon <= 1.0:
            self.epsilon = epsilon
        else:
            raise ValueError("epsilon must be between 0 and 1")

    def epsilon_greedy(self):
        if np.random.random_sample() < self.epsilon:
            action = np.random.choice(self.k)
        else:
            _max = self.Q.max()
            max_list = np.where(self.Q == _max)[0]
            action = np.random.choice(max_list)
        return action

    def updateQ(self, action):
        self.N[action] += 1
        reward = np.random.normal(self.reward[action], 1, 1)
        self.Q[action]+= (reward - self.Q[action])/self.N[action]
        return reward

    def pull_lever_once(self):
        action = self.epsilon_greedy()
        return self.updateQ(action)

    def pull_lever(self, num_pull=1000):
        for pull in range(num_pull):
            self.pull_lever_once()


class NonStationaryBandit(MultiArmBandit):
    def __init__(self, constant_alpha=True, **kwargs):
        super().__init__(**kwargs)
        self.constant_alpha = constant_alpha

    def updateQ(self, action):
        self.N[action] += 1
        reward_delta = np.random.normal(0, 0.1, self.k)
        self.reward += reward_delta
        reward = np.random.normal(self.reward[action], 1, 1)
        if self.constant_alpha:
            self.Q[action] += (reward - self.Q[action])*0.1
        else:
            self.Q[action] += (reward - self.Q[action])/self.N[action]
        return reward


class OptimisticInitialValueBandit(MultiArmBandit):
    def __init__(self, optimistic=True, **kwargs):
        self.optimistic = optimistic
        super().__init__(**kwargs)

    def initQ(self):
        initial_value = 10.0 if self.optimistic else 0.0
        self.Q = np.array([initial_value]*self.k)
        self.N = [0]*self.k

    def updateQ(self, action):
        self.N[action] += 1
        reward = np.random.normal(self.reward[action], 1, 1)
        self.Q[action] += (reward - self.Q[action])*0.1
        return reward


class UCB_Bandit(MultiArmBandit):
    def __init__(self, ucb=True, c=2, **kwargs):
        super().__init__(**kwargs)
        self.total_step = 0
        self.ucb = ucb
        self.c = 2

    def ucb_formula(self, action):
        """upper confidence bound with smoothing"""
        upper_bound = self.Q[action] + self.c*np.sqrt(np.log(self.total_step+1)/(self.N[action]+1))
        return upper_bound

    def ucb_argmax(self):
        upper_bounds = [self.ucb_formula(action)for action in range(self.k)]
        _max = np.array(upper_bounds).max()
        max_list = np.where(upper_bounds == _max)[0]
        action = np.random.choice(max_list)
        return action

    def updateQ(self, action):
        self.N[action] += 1
        self.total_step += 1
        reward = np.random.normal(self.reward[action], 1, 1)
        self.Q[action] += (reward - self.Q[action])/self.N[action]
        return reward

    def pull_lever_once(self):
        if self.ucb:
            action = self.ucb_argmax()
        else:
            action = self.epsilon_greedy()
        return self.updateQ(action)
