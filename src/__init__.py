

from .bandits import (MultiArmBandit,
                         NonStationaryBandit,
                         OptimisticInitialValueBandit,
                         UCB_Bandit)

from .simulations import (Simulation,
                             NonStationarySimulation,
                             OptimisticInitialValueSimulation,
                             UCB_Simulation)

from .plot import (plot_epsilon_greedy,
                      plot_nonstationary,
                      plot_optimistic_initial_value,
                      plot_upper_confidence_bound)


__all__ = ["MultiArmBandit",
           "NonStationaryBandit",
           "OptimisticInitialValueBandit",
           "UCB_Bandit",
           "Simulation",
           "NonStationarySimulation",
           "OptimisticInitialValueSimulation",
           "UCB_Simulation",
           "plot_epsilon_greedy",
           "plot_nonstationary",
           "plot_optimistic_initial_value",
           "plot_upper_confidence_bound",
           ]
