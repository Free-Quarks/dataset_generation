import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# Function to define the differential equations

def serid_model(y, t, beta, gamma, N):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dIdt = gamma * E - D * I
    dRdt = (1 - D) * gamma * I
    dDdt = D * gamma * I
    return dSdt, dEdt, dIdt, dRdt, dDdt

# Function to simulate and plot the model

def simulate_serid_model(beta, gamma, D):
    N = 1000
    I0 = 1
    E0 = 0
    S0 = N - I0 - E0
    R0 = 0
    D0 = 0
    y0 = S0, E0, I0, R0, D0
    t = np.linspace(0, 100, 100)
    sol = odeint(serid_model, y0, t, args=(beta, gamma, N))
    S, E, I, R, D = sol.T
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SERID Model Simulation')
    plt.show()

simulate_serid_model(0.3, 0.1, 0.05)
