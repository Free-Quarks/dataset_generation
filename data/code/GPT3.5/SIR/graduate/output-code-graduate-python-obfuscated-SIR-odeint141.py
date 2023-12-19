import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plot_SIR(S, I, R, t):
    plt.figure(figsize=(10,6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()

N = 1000
beta = 0.2
D = 10
gamma = 1 / D
S0, I0, R0 = N-1, 1, 0
y0 = S0, I0, R0

t = np.linspace(0, 100, 100)

solution = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = solution.T

plot_SIR(S, I, R, t)

