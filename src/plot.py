import matplotlib.pyplot as plt

from .simulations import (Simulation, NonStationarySimulation,
                          OptimisticInitialValueSimulation,
                          UCB_Simulation)


def plot_epsilon_greedy(num_pull=1000):
    f, ax = plt.subplots()
    epsilons = [0.1, 0.01, 0.0]
    for eps in epsilons:
        sim = Simulation(epsilon=eps, num_sim=2000,num_pull=num_pull)
        historical_average = sim.run()
        ax.plot(historical_average)
        plt.title("Exploration vs Exploitation")
        plt.xlabel("Number of Steps")
        plt.ylabel("Average rewards over 2000 simulations")
        plt.legend(("Epsilon=0.1", "Epsilon=0.01", "Epsilon=0.0"))
    plt.show()


def plot_nonstationary():
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


def plot_optimistic_initial_value():
    f, ax = plt.subplots()
    epsilons = [0.0, 0.10]
    optimistic = [True, False]
    for i in range(2):
        sim = OptimisticInitialValueSimulation(                                        
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


def plot_upper_confidence_bound():
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


if __name__ == "__main__":
    plot_epsilon_greedy()
    # plot_nonstationary()
    # plot_optimistic_initial_value()
    # plot_upper_confidence_bound()
