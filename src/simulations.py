import numpy as np
from .bandits import (MultiArmBandit, NonStationaryBandit,
                      OptimisticInitialValueBandit, UCB_Bandit)


class Simulation:
    def __init__(self, bandit=MultiArmBandit, num_sim=2000, num_pull=1000,
                 epsilon=0.1):
        self.bandit = bandit
        self.num_sim = num_sim
        self.num_pull = num_pull
        self.epsilon = epsilon

    def init_bandit(self, **kwargs):
        return self.bandit(epsilon=self.epsilon, **kwargs)

    def run(self):
        historical_average = np.zeros(self.num_pull)
        for sim in range(self.num_sim):
            if sim % 100 == 0:
                print(sim)
            bandit = self.init_bandit()
            for i in range(self.num_pull):
                reward = bandit.pull_lever_once()
                historical_average[i] += reward
        return historical_average/self.num_sim


class NonStationarySimulation(Simulation):
    def __init__(self,
                 constant_alpha=True,
                 bandit=NonStationaryBandit, **kwargs):
        super().__init__(**kwargs)
        self.constant_alpha = constant_alpha
        self.bandit = bandit

    def init_bandit(self, **kwargs):
        return self.bandit(epsilon=self.epsilon,
                           constant_alpha=self.constant_alpha, **kwargs)


class OptimisticInitialValueSimulation(Simulation):
    def __init__(self, bandit=OptimisticInitialValueBandit,
                 optimistic=True, **kwargs):
        super().__init__(**kwargs)
        self.bandit = bandit
        self.optimistic = optimistic

    def init_bandit(self, **kwargs):
        return self.bandit(epsilon=self.epsilon,
                           optimistic=self.optimistic, **kwargs)


class UCB_Simulation(Simulation):
    def __init__(self, bandit=UCB_Bandit, ucb=True, **kwargs):
        super().__init__(**kwargs)
        self.bandit = bandit
        self.ucb = ucb

    def init_bandit(self, **kwargs):
        return self.bandit(epsilon=self.epsilon,
                           ucb=self.ucb, **kwargs)
