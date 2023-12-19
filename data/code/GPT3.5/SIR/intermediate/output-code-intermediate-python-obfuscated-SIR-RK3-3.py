```python
import numpy as np
import matplotlib.pyplot as plt

# Function to define the SIR model

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to solve the SIR model using RK3

def solve_sir_model(beta, gamma, t0, t_end, S0, I0, R0, N, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), 3))
    y[0] = [S0, I0, R0]
    for i in range(1, len(t)):
        k1 = h * sir_model(y[i-1], t[i-1], beta, gamma)
        k2 = h * sir_model(y[i-1] + k1/2, t[i-1] + h/2, beta, gamma)
        k3 = h * sir_model(y[i-1] - k1 + 2*k2, t[i-1] + h, beta, gamma)
        y[i] = y[i-1] + (k1 + 4*k2 + k3) / 6
    return t, y

# Parameters
beta = 0.2
gamma = 0.1
t0 = 0
t_end = 100
S0 = 0.99
I0 = 0.01
R0 = 0
N = 1000
h = 0.1

# Solve SIR model
t, y = solve_sir_model(beta, gamma, t0, t_end, S0, I0, R0, N, h)

# Plot results
plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Infected')
plt.plot(t, y[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
```
