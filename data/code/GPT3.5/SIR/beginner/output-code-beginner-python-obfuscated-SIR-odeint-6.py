import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that defines the SIR model
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Function to simulate and plot the SIR model
def simulate_sir_model(N, I0, R0, beta, gamma, T):
    S0 = N - I0 - R0
    t = np.linspace(0, T, T+1)
    y0 = S0, I0, R0
    solution = odeint(sir_model, y0, t, args=(N, beta, gamma))
    S, I, R = solution.T
    
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.title('SIR Model Simulation')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
}

