```python
import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, N, I0, R0, days):
    # SIR model
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.linspace(0, days, days)

    # Integrate the SIR equations over time
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    # Plot the data
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model using RK4')
    plt.legend()
    plt.grid(True)
    plt.show()


# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0, R0 = 1, 0

# Run the SIR model
SIR_RK4(beta, gamma, N, I0, R0, days=160)
```
