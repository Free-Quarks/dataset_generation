import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(S0, I0, R0, beta, gamma, T, dt):
    S, I, R = [S0], [I0], [R0]
    t = np.arange(0, T+dt, dt)
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_I = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    return np.stack([S, I, R]).T

def plot_SIR(SIR_data):
    plt.figure(figsize=(12, 8))
    plt.plot(SIR_data[:,0], label='Susceptible')
    plt.plot(SIR_data[:,1], label='Infected')
    plt.plot(SIR_data[:,2], label='Recovered')
    plt.legend()
    plt.show()

S0, I0, R0 = 999, 1, 0
beta, gamma = 0.2, 0.1
T, dt = 160, 0.1

SIR_data = SIR_model(S0, I0, R0, beta, gamma, T, dt)
plot_SIR(SIR_data)
