# Reinforcement Learning: Multi-Arm Bandit

### Getting Started

#### Installation

```bash
$ conda env create -f environment.yml
$ conda activate bandits
```

#### General Usage

```python
(bandits) ~/MultiArmBandit$ python
>>> from src import *
>>> plot_epsilon_greedy()
>>> plot_nonstationary()
>>> plot_optimistic_initial_value()
>>> plot_upper_confidence_bound()
```

### Exploration vs Exploitation: changing the value of epsilon  
    


`plot_epsilon_greedy()` 

1000 steps


<img src="image/epsilon_compare.png" width="1000" height="400" />

`plot_epsilon_greedy(10000)`  
10000 steps


<img src="image/epsilon_compare10000steps.png" width="1000" height="400" />

---
### Non-stationary reward

`plot_nonstationary()` 


<img src="image/non_stationary.png" width="1000" height="400" />

---
### Optimistic initial value
`plot_optimistic_initial_value()`  

<img src="image/optimistic_initial_value.png" width="1000" height="400" />

---
### Upper Confidence Bound

`plot_upper_confidence_bound()`

<img src="image/upper_confidence_bound.png" width="1000" height="400" />
