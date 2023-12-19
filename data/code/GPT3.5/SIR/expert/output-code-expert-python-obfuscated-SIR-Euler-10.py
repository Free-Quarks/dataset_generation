```python
import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, population, infected, recovered):
    time_steps = np.arange(0, 100, 0.1)
    susceptible = np.zeros(len(time_steps))
    infected = np.zeros(len(time_steps))
    recovered = np.zeros(len(time_steps))

    susceptible[0] = population - infected[0] - recovered[0]

    for i in range(len(time_steps)-1):
        susceptible[i+1] = susceptible[i] - (beta * infected[i] * susceptible[i])
        infected[i+1] = infected[i] + (beta * infected[i] * susceptible[i]) - (gamma * infected[i])
        recovered[i+1] = recovered[i] + (gamma * infected[i])

    plt.plot(time_steps, susceptible, label='Susceptible')
    plt.plot(time_steps, infected, label='Infected')
    plt.plot(time_steps, recovered, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

sir_model(0.25, 0.1, 1000, 1, 0)
```
