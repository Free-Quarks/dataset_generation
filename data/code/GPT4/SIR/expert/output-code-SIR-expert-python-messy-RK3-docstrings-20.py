import numpy as np
import matplotlib.pyplot as plt
import json

def RK3_SIR(beta, gamma, S0, I0, R0, t):
    """
    This function simulates the SIR model using the RK3 method.
    The SIR model is a simple mathematical model to understand outbreak of diseases.
    :param beta: the effective contact rate per day
    :param gamma: the recovery rate per day
    :param S0: the initial susceptible population
    :param I0: the initial infected population
    :param R0: the initial recovered population
    :param t: the time steps
    :return: S, I, R
    """
    N = S0 + I0 + R0 
    dt = 1.0
    S, I, R = [S0], [I0], [R0]
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1]/N)*dt
        next_I = I[-1] + (beta*S[-1]*I[-1]/N - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
    return S, I, R

# test the function
S0, I0, R0 = 999, 1, 0
beta, gamma = 0.2, 0.1
t = np.linspace(0, 160, 160)
S, I, R = RK3_SIR(beta, gamma, S0, I0, R0, t)

plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.legend()
plt.show()
