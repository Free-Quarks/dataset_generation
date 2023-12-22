import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(S0, I0, R0, beta, gamma, t):
    S, I, R = [S0], [I0], [R0]
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - beta*S[-1]*I[-1]*dt
        next_I = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        next_R = R[-1] + gamma*I[-1]*dt
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, I, R]).T

# Parameters
S0, I0, R0 = 990, 10, 0  # initial conditions: ten infected, rest susceptible
beta = 0.5  # infection rate
gamma = 0.1  # recovery rate
t = np.linspace(0, 200, 200)  # time grid

# Run simulation
results = SIR_model(S0, I0, R0, beta, gamma, t)

# Plot results
plt.plot(t, results)
plt.legend(['Susceptible', 'Infected', 'Recovered'])
plt.show()
