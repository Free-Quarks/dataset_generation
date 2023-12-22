from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import json

def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plot_sir(t, S, I, R):
    plt.figure(figsize=(6,4))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time /days')
    plt.ylabel('Number (1000s)')
    plt.grid()
    plt.legend()
    plt.show()
