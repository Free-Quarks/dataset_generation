import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def run_SIR_model(S, I, R, beta, gamma, t):
    y0 = S, I, R
    args = (beta, gamma)
    solution = odeint(SIR_model, y0, t, args)
    S_sol, I_sol, R_sol = solution[:, 0], solution[:, 1], solution[:, 2]
    return S_sol, I_sol, R_sol


def plot_SIR_model(S_sol, I_sol, R_sol, t):
    plt.plot(t, S_sol, 'b-', label='Susceptible')
    plt.plot(t, I_sol, 'r-', label='Infected')
    plt.plot(t, R_sol, 'g-', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

