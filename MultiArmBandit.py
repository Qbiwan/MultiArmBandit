import numpy as np
import matplotlib.pyplot as plt


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


class Simulation:
    def __init__(self, bandit=MultiArmBandit, num_sim=2000, num_pull=1000,
                 epsilon=0.1):
        self.bandit = MultiArmBandit
        self.num_sim = num_sim
        self.num_pull = num_pull
        self.epsilon = epsilon

    def run(self):
        historical_average = np.zeros(self.num_pull)
        for sim in range(self.num_sim):
            bandit = self.bandit(epsilon=self.epsilon)
            for i in range(self.num_pull):
                reward = bandit.pull_lever_once()
                historical_average[i] += reward
        return historical_average/self.num_sim


if __name__ == "__main__":

    bandit = MultiArmBandit()
    bandit.pull_lever(100000)
    print(f"True Rewards : \n {bandit.reward} \n")
    print(f"Q : \n {bandit.Q} \n")
    print(f"N : \n {bandit.N} \n")

    f, ax = plt.subplots()
    epsilons = [0.1, 0.01, 0.0]
    for eps in epsilons:
        sim = Simulation(epsilon=eps, num_sim=2000)
        historical_average = sim.run()
        ax.plot(historical_average)
        plt.title("Exploration vs Exploitation")
        plt.xlabel("Number of Steps")
        plt.ylabel("Average rewards over 2000 simulations")
        plt.legend(("Epsilon=0.1", "Epsilon=0.01", "Epsilon=0.0"))
    plt.show()