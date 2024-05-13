```python
import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    def derivs(y, t, N, beta, gamma, sigma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Initial conditions vector
    y0 = N - I0 - E0 - R0

    # A grid of time points (in days)
    t = np.linspace(0, T, T)

    # Integrate the SIR equations over the time grid, t
    ret = odeint(derivs, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = ret.T

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, E, 'y', label='Exposed')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_ylim(0, N)
    ax.set_title('SEIR Model')
    ax.legend()
    plt.show()


# Parameters
beta = 0.2
gamma = 0.1
sigma = 0.05
N = 100000
I0 = 1
E0 = 0
R0 = 0
T = 300

seir_model(beta, gamma, sigma, N, I0, E0, R0, T)
```
