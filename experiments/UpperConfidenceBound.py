import matplotlib.pyplot as plt
import numpy as np

from MultiArmBandit import MultiArmBandit, Simulation


class UCB_MultiArmBandit(MultiArmBandit):
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


class UCB_Simulation(Simulation):
    def __init__(self, bandit=UCB_MultiArmBandit, ucb=True, **kwargs):
        super().__init__(**kwargs)
        self.bandit = bandit
        self.ucb = ucb

    def run(self):
        historical_average = np.zeros(self.num_pull)
        for sim in range(self.num_sim):
            bandit = self.bandit(epsilon=self.epsilon,
                                 ucb=self.ucb)
            for i in range(self.num_pull):
                reward = bandit.pull_lever_once()
                historical_average[i] += reward
        return historical_average/self.num_sim


if __name__ == "__main__":
    f, ax = plt.subplots()
    eps = [0.0, 0.1]
    ucb = [True, False]
    for i in range(2):
        sim = UCB_Simulation(epsilon=eps[i], num_sim=2000, ucb=ucb[i])
        historical_average = sim.run()
        ax.plot(historical_average)
        plt.title("Upper Confidence Bound vs Epsilon-Greedy")
        plt.xlabel("Number of Steps")
        plt.ylabel("Average rewards over 2000 simulations")
        plt.legend(("UCB: Epsilon=0.0", "Epsilon-greedy: Epsilon=0.1"))
    plt.show()
