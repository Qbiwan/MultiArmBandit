"""Multi-arm Bandits

Usage:
    Each bandit class seeks rewards with a different strategy.
"""

import numpy as np


class MultiArmBandit:
    """A basic multi-arm bandit.

    Parameters:
    ----------
        k : int, default= 10
            Specify the number of arms that the multi-arm bandit has and that
            can be pulled to receive rewards.
        epilson : float, default=0.1
            The epsilon in epsilon-greedy. If epsilon is 0.1, the bandit would
            implement random exploration 10% of the time, and would take the
            greedy action to maximise reward 90% of the time.
            Epsilon lies between 0.0 and 1.0

    Attributes:
    ----------
        rewards : a vector of numpy random normal of size k
            Represents the true reward of each of the k number of arms.
            The actual rewards would fluctuate around this mean, and the manner
            in which it fluctuates would change depending on which bandit class
            is being used.

        Q : a vector of floats of size k
            The average value of all rewards received hitherto by each arm.
            It converges to the true reward in the long run, unless one is
            using constant alpha in a non-stationary reward environment.

        N : a vector of floats of size k
            The number of pulls received by each arm. The default total number
            of pulls is 1000, and each arm would receive a portion of these
            pulls, depending on the strategy adopted by the bandit.

    """

    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.reward = np.random.randn(k)
        self.Q = None
        self.N = None
        self.initQ()
        self.epsilon = epsilon

    def initQ(self):
        """Initialize Q-values

        Initializes Q, a vector of size k to be the starting Q-values for
        each arm, and N, a vector of size k to be the counter for the number
        of pulls for each arm

        """
        self.Q = np.array([0.0]*self.k)
        self.N = [0]*self.k

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """Prevents epsilon from being set outside range"""
        if not (0.0 <= value <= 1.0):
            raise ValueError("epsilon must be between 0 and 1")
        self._epsilon = value

    def epsilon_greedy(self):
        """Implements epsilon-greedy strategy

        Takes an action randomly x% of the time and an action that maximizes
        current Q-value (1-x)% of the time. If there are several candidates
        for maximum Q-values, a random tie-breaker is called.

        Returns:
            action:
                The arm to be pulled under epsilon-greedy strategy
        """
        if np.random.random_sample() < self.epsilon:
            action = np.random.choice(self.k)
        else:
            _max = self.Q.max()
            max_list = np.where(self.Q == _max)[0]
            action = np.random.choice(max_list)
        return action

    def updateQ(self, action):
        """Update Q-value with the latest reward

        Increments N by one to record the total number of pulls for the arm
        corresponding to action.
        Actual reward is a normally-distributed random variable with standard
        deviation of 1 and mean set the true reward value of this arm.
        Q-value is updated using the simple moving average.[1]

        Parameters:
        ----------
            action:
                the specific arm to be pulled, out of all k arms.

        Returns:
        -------
            rewards:
                the actual reward received from pulling the specific arm.

        References:
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moving_average   

        """
        self.N[action] += 1
        reward = np.random.normal(self.reward[action], 1, 1)
        self.Q[action] += (reward - self.Q[action])/self.N[action]
        return reward

    def pull_lever_once(self):
        """Pull once a specifc arm as chosen by epsilon-greedy strategy"""
        action = self.epsilon_greedy()
        return self.updateQ(action)

    def pull_lever(self, num_pull=1000):
        """Select arm using a specific strategy, and pull by default 1000 times"""
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
