import matplotlib.pyplot as plt
import numpy as np

from MultiArmBandit import MultiArmBandit, Simulation


class NonStationaryMultiArmBandit(MultiArmBandit):
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


class NonStationarySimulation(Simulation):
    def __init__(self,
                 constant_alpha=True,
                 bandit=NonStationaryMultiArmBandit, **kwargs):
        super().__init__(**kwargs)
        self.constant_alpha = constant_alpha
        self.bandit = bandit

    def run(self):
        historical_average = np.zeros(self.num_pull)
        for sim in range(self.num_sim):
            if sim % 100 == 0:
                print(sim)
            bandit = self.bandit(epsilon=self.epsilon,
                                 constant_alpha=self.constant_alpha
                                 )
            for i in range(self.num_pull):
                reward = bandit.pull_lever_once()
                historical_average[i] += reward
        return historical_average/self.num_sim


if __name__ == "__main__":
    f, ax = plt.subplots()
    constant_alphas = [True, False]
    for constant_alpha in constant_alphas:
        sim = NonStationarySimulation(epsilon=0.1,
                                      num_sim=2000,
                                      num_pull=10000,
                                      constant_alpha=constant_alpha
                                      )
        historical_average = sim.run()
        ax.plot(historical_average)
        plt.title("Constant alpha vs Sample average")
        plt.xlabel("Number of Steps")
        plt.ylabel("Average rewards over 2000 simulations")
        plt.legend(("Constant alpha", "Sample average"))
    plt.show()
