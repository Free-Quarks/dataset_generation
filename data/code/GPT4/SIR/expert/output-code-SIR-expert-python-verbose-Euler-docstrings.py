import numpy as np
import matplotlib.pyplot as plt
import json

def Euler_SIR(S0, I0, R0, beta, gamma, T, dt):
    """
    Simulate the SIR model using the Euler method.

    Parameters:
    S0, I0, R0: initial values for S, I, R
    beta: infection rate
    gamma: recovery rate
    T: total time to simulate
    dt: time step for simulation

    Returns:
    t: an array of time points 
    S, I, R: arrays of the corresponding S, I, R values at each time point
    """

    N = S0 + I0 + R0
    t = np.arange(0, T, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        S[i] = S[i-1] - dt*beta*S[i-1]*I[i-1]/N
        I[i] = I[i-1] + dt*(beta*S[i-1]*I[i-1]/N - gamma*I[i-1])
        R[i] = R[i-1] + dt*gamma*I[i-1]

    return t, S, I, R

# Use the function
t, S, I, R = Euler_SIR(999, 1, 0, 0.1, 0.05, 100, 0.1)

# Plot the results
plt.figure()
plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.legend()
plt.show()
