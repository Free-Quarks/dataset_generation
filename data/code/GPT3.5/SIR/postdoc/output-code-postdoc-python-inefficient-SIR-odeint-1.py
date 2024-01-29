import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def simulate_SIR_model(N, beta, gamma, I0, R0, t_max):
    S0 = N - I0 - R0
    t = np.linspace(0, t_max, t_max+1)
    y0 = S0, I0, R0
    solution = odeint(SIR_model, y0, t, args=(N, beta, gamma))
    S, I, R = solution.T
    
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()
    

simulate_SIR_model(N=1000, beta=0.2, gamma=0.1, I0=1, R0=0, t_max=100)
