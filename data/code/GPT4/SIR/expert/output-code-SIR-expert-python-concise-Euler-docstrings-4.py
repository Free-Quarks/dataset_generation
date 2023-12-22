import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(init_conditions, params, t):
    """
    Simulate the SIR model using Euler's method.

    Parameters:
    init_conditions (tuple): initial conditions (S0, I0, R0)
    params (tuple): model parameters (beta, gamma)
    t (int): time period

    Returns:
    SIR (ndarray): model simulation results
    """
    S0, I0, R0 = init_conditions
    S, I, R = [S0], [I0], [R0]
    beta, gamma = params
    dt = 1.0
    for _ in range(t):
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_I = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    return np.stack([S, I, R]).T

# Plot the results
t = 100
init_conditions = (0.99, 0.01, 0.0)
params = (0.35, 0.1)

SIR = sir_model_euler(init_conditions, params, t)

plt.plot(SIR)
plt.show()
