import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(S0, I0, R0, beta, gamma, T, dt):
    """
    The SIR model differential equations.
    S0: Initial susceptible population
    I0: Initial infected population
    R0: Initial recovered population
    beta: Contact rate
    gamma: Recovery rate
    T: Time
    dt: Time step size
    """
    N = S0 + I0 + R0
    S, I, R = [S0], [I0], [R0]
    t = np.arange(0, T, dt)
    
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1]/N)*dt
        next_I = I[-1] + (beta*S[-1]*I[-1]/N - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1]*dt)
        
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
        
    return t, S, I, R

t, S, I, R = sir_model(S0=990, I0=10, R0=0, beta=0.5, gamma=0.1, T=100, dt=0.1)
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.legend()
plt.show()
