import matplotlib.pyplot as plt
import numpy as np

from MultiArmBandit import MultiArmBandit, Simulation


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


class OptimisticInitialValueSimulation(Simulation):
    def __init__(self, bandit=OptimisticInitialValueBandit,
                 optimistic=True, **kwargs):
        super().__init__(**kwargs)
        self.bandit = bandit
        self.optimistic = optimistic

    def run(self):
        historical_average = np.zeros(self.num_pull)
        for sim in range(self.num_sim):
            if sim % 500 == 0:
                print(sim)
            bandit = self.bandit(optimistic=self.optimistic)
            for i in range(self.num_pull):
                reward = bandit.pull_lever_once()
                historical_average[i] += reward
        return historical_average/self.num_sim


if __name__ == "__main__":
    f, ax = plt.subplots()
    epsilons = [0.0, 0.10]
    optimistic = [True, False]
    for i in range(2):
        sim = OptimisticInitialValueSimulation(
                                        bandit=OptimisticInitialValueBandit,
                                        epsilon=epsilons[i],
                                        optimistic=optimistic[i],          
                                        num_sim=2000
                                              )
        historical_average = sim.run()
        ax.plot(historical_average)
        plt.title("Inititial Value: Optimistic Vs Realistic")
        plt.xlabel("Number of Steps")
        plt.ylabel("Average rewards over 2000 simulations")
        plt.legend(("Optimistic:epsilon=0.0,initial value=10.0",
                    "Realistic:epsilon=0.1, initial value=0.0"))
    plt.show()
